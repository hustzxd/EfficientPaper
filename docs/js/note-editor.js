// Note Editor for MkDocs
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
    // Try different URL patterns:
    // - /notes/2025/07NWF4VE/ (with trailing slash)
    // - /notes/2025/07NWF4VE (without trailing slash)
    const pathname = window.location.pathname;

    // Pattern 1: with trailing slash
    let pathMatch = pathname.match(/\/notes\/(\d+)\/([^\/]+)\//);
    if (pathMatch) {
      return `notes/${pathMatch[1]}/${pathMatch[2]}/note.md`;
    }

    // Pattern 2: without trailing slash (ends with paper ID)
    pathMatch = pathname.match(/\/notes\/(\d+)\/([^\/]+)$/);
    if (pathMatch) {
      return `notes/${pathMatch[1]}/${pathMatch[2]}/note.md`;
    }

    return null;
  }

  // Create editor UI
  function createEditorUI() {
    // Create edit button
    const editBtn = document.createElement('button');
    editBtn.id = 'note-edit-btn';
    editBtn.className = 'note-edit-btn';
    editBtn.innerHTML = '✏️ Edit Note';
    editBtn.title = 'Edit this note (requires local server)';

    // Create editor modal
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
          <textarea id="note-editor-textarea" placeholder="Write your note in Markdown..."></textarea>
        </div>
        <div class="note-editor-footer">
          <span class="note-path"></span>
          <span id="note-save-status"></span>
        </div>
      </div>
    `;

    // Insert UI elements - try multiple possible locations
    document.body.appendChild(editBtn);
    document.body.appendChild(editorModal);

    return { editBtn, editorModal };
  }

  // Load note content
  async function loadNoteContent(notePath) {
    try {
      // Use the API server to load the raw markdown file
      const response = await fetch(`http://localhost:8001/api/load-note?path=${encodeURIComponent(notePath)}`);

      if (response.ok) {
        const data = await response.json();

        if (data.exists) {
          return data.content;
        }
      }

      // If file doesn't exist or API fails, return template
      return `# ${document.querySelector('h1')?.textContent || 'Paper Note'}\n\n## Abstract\n\n## Key Points\n\n## Methodology\n\n## Results\n\n## Conclusion\n\n`;
    } catch (error) {
      console.error('Failed to load note:', error);
      // Return template on error
      return `# ${document.querySelector('h1')?.textContent || 'Paper Note'}\n\n## Abstract\n\n## Key Points\n\n## Methodology\n\n## Results\n\n## Conclusion\n\n`;
    }
  }

  // Save note content
  async function saveNoteContent(notePath, content) {
    try {
      const response = await fetch('http://localhost:8001/save_note', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          path: notePath,
          content: content
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Failed to save note:', error);
      throw error;
    }
  }

  // Initialize editor
  async function initEditor() {
    console.log('[Note Editor] Initializing on path:', window.location.pathname);

    // Check if local server is available first
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

    console.log('[Note Editor] UI created, button visible:', editBtn.offsetParent !== null);

    // Open editor
    editBtn.addEventListener('click', async () => {
      console.log('[Note Editor] Opening editor');
      const content = await loadNoteContent(notePath);
      textarea.value = content;
      editorModal.style.display = 'flex';
      textarea.focus();
      saveStatus.textContent = '';
    });

    // Close editor
    function closeEditor() {
      editorModal.style.display = 'none';
      saveStatus.textContent = '';
    }

    cancelBtn.addEventListener('click', closeEditor);

    // Close on overlay click
    editorModal.addEventListener('click', (e) => {
      if (e.target === editorModal) {
        closeEditor();
      }
    });

    // Close on ESC key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && editorModal.style.display === 'flex') {
        closeEditor();
      }
    });

    // Save note
    saveBtn.addEventListener('click', async () => {
      const content = textarea.value;
      saveStatus.textContent = 'Saving...';
      saveStatus.className = 'status-saving';
      saveBtn.disabled = true;

      try {
        await saveNoteContent(notePath, content);
        saveStatus.textContent = '✓ Saved! Auto-reload running, refresh browser to see changes.';
        saveStatus.className = 'status-success';

        // Re-enable save button after a short delay
        setTimeout(() => {
          saveBtn.disabled = false;
        }, 2000);
      } catch (error) {
        saveStatus.textContent = '✗ Failed to save. Make sure the server is running.';
        saveStatus.className = 'status-error';
        saveBtn.disabled = false;
      }
    });

    // Auto-resize textarea
    textarea.addEventListener('input', () => {
      textarea.style.height = 'auto';
      textarea.style.height = textarea.scrollHeight + 'px';
    });
  }

  // Wait for DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initEditor);
  } else {
    initEditor();
  }
})();
