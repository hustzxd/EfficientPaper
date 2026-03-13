// Note Editor for MkDocs (EasyMDE version)
(function() {
  'use strict';

  // Only enable on note pages
  const isNotePage = window.location.pathname.includes('/notes/');
  if (!isNotePage) return;

  // Check if local server is available
  async function checkLocalServer() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);
      const response = await fetch('http://localhost:8001/api/get-keywords', {
        method: 'GET',
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      return response.ok;
    } catch (e) {
      return false;
    }
  }

  // Get note path from URL
  function getNotePath() {
    const pathname = window.location.pathname;

    let pathMatch = pathname.match(/\/notes\/(\d+)\/([^\/]+)\//);
    if (pathMatch) {
      return `notes/${pathMatch[1]}/${pathMatch[2]}/note.md`;
    }

    pathMatch = pathname.match(/\/notes\/(\d+)\/([^\/]+)$/);
    if (pathMatch) {
      return `notes/${pathMatch[1]}/${pathMatch[2]}/note.md`;
    }

    return null;
  }

  // Create editor UI
  function createEditorUI() {
    const editBtn = document.createElement('button');
    editBtn.id = 'note-edit-btn';
    editBtn.className = 'note-edit-btn';
    editBtn.innerHTML = '✏️ Edit Note';
    editBtn.title = 'Edit this note (requires local server)';

    const editorModal = document.createElement('div');
    editorModal.id = 'note-editor-modal';
    editorModal.className = 'note-editor-modal';
    editorModal.style.display = 'none';
    editorModal.innerHTML = `
      <div class="note-editor-container">
        <div class="note-editor-header">
          <h2>Edit Note</h2>
          <div class="note-editor-actions">
            <button id="note-save-btn" class="btn-primary">💾 Save</button>
            <button id="note-cancel-btn" class="btn-secondary">Cancel</button>
          </div>
        </div>
        <div class="note-editor-body">
          <textarea id="note-editor-textarea"></textarea>
        </div>
        <div class="note-editor-footer">
          <span class="note-path"></span>
          <span id="note-save-status"></span>
        </div>
      </div>
    `;

    document.body.appendChild(editBtn);
    document.body.appendChild(editorModal);

    return { editBtn, editorModal };
  }

  // Load note content
  async function loadNoteContent(notePath) {
    try {
      const response = await fetch(`http://localhost:8001/api/load-note?path=${encodeURIComponent(notePath)}`);
      if (response.ok) {
        const data = await response.json();
        if (data.exists) {
          return data.content;
        }
      }
      return `# ${document.querySelector('h1')?.textContent || 'Paper Note'}\n\n## Abstract\n\n## Key Points\n\n## Methodology\n\n## Results\n\n## Conclusion\n\n`;
    } catch (error) {
      console.error('Failed to load note:', error);
      return `# ${document.querySelector('h1')?.textContent || 'Paper Note'}\n\n## Abstract\n\n## Key Points\n\n## Methodology\n\n## Results\n\n## Conclusion\n\n`;
    }
  }

  // Save note content
  async function saveNoteContent(notePath, content) {
    try {
      const response = await fetch('http://localhost:8001/save_note', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: notePath, content: content })
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Failed to save note:', error);
      throw error;
    }
  }

  // Initialize editor
  async function initEditor() {
    console.log('[Note Editor] Initializing on path:', window.location.pathname);

    const serverAvailable = await checkLocalServer();
    if (!serverAvailable) {
      console.log('[Note Editor] Local server not available, hiding edit button');
      return;
    }

    const notePath = getNotePath();
    if (!notePath) {
      console.warn('[Note Editor] Could not determine note path from URL:', window.location.pathname);
      return;
    }

    console.log('[Note Editor] Note path detected:', notePath);

    const { editBtn, editorModal } = createEditorUI();
    const textarea = document.getElementById('note-editor-textarea');
    const saveBtn = document.getElementById('note-save-btn');
    const cancelBtn = document.getElementById('note-cancel-btn');
    const saveStatus = document.getElementById('note-save-status');
    const pathDisplay = editorModal.querySelector('.note-path');

    pathDisplay.textContent = notePath;

    let easyMDE = null;

    // Open editor
    editBtn.addEventListener('click', async () => {
      console.log('[Note Editor] Opening editor');
      const content = await loadNoteContent(notePath);

      editorModal.style.display = 'flex';
      saveStatus.textContent = '';

      // Initialize EasyMDE after modal is visible (needed for correct rendering)
      if (easyMDE) {
        easyMDE.toTextArea();
        easyMDE = null;
      }

      textarea.value = content;

      easyMDE = new EasyMDE({
        element: textarea,
        autofocus: true,
        spellChecker: false,
        status: ['lines', 'words', 'cursor'],
        minHeight: '400px',
        maxHeight: '60vh',
        toolbar: [
          'bold', 'italic', 'strikethrough', 'heading', '|',
          'code', 'quote', 'unordered-list', 'ordered-list', '|',
          'link', 'image', 'table', 'horizontal-rule', '|',
          'preview', 'side-by-side', 'fullscreen', '|',
          'undo', 'redo', '|',
          'guide'
        ],
        previewImagesInEditor: true,
        renderingConfig: {
          codeSyntaxHighlighting: true
        },
        shortcuts: {
          toggleSideBySide: 'Cmd-P',
          togglePreview: 'Cmd-Shift-P'
        }
      });
    });

    // Close editor
    function closeEditor() {
      if (easyMDE) {
        easyMDE.toTextArea();
        easyMDE = null;
      }
      editorModal.style.display = 'none';
      saveStatus.textContent = '';
    }

    cancelBtn.addEventListener('click', closeEditor);

    editorModal.addEventListener('click', (e) => {
      if (e.target === editorModal) {
        closeEditor();
      }
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && editorModal.style.display === 'flex') {
        // Don't close if in fullscreen mode
        if (easyMDE && easyMDE.isFullscreenActive()) return;
        closeEditor();
      }
    });

    // Save note
    saveBtn.addEventListener('click', async () => {
      const content = easyMDE ? easyMDE.value() : textarea.value;
      saveStatus.textContent = 'Saving...';
      saveStatus.className = 'status-saving';
      saveBtn.disabled = true;

      try {
        await saveNoteContent(notePath, content);
        saveStatus.textContent = '✓ Saved!';
        saveStatus.className = 'status-success';
        setTimeout(() => { saveBtn.disabled = false; }, 2000);
      } catch (error) {
        saveStatus.textContent = '✗ Failed to save. Make sure the server is running.';
        saveStatus.className = 'status-error';
        saveBtn.disabled = false;
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initEditor);
  } else {
    initEditor();
  }
})();
