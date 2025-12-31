# Efficient Paper

<div id="paper-search-app">
  <div class="search-header">
    <div class="search-box">
      <input type="text" id="search-input" placeholder="Search papers by title, author, institution..." autofocus>
      <span class="search-hint">Press <kbd>/</kbd> to focus, <kbd>Esc</kbd> to clear</span>
    </div>
    <div class="filter-bar">
      <select id="year-filter">
        <option value="">All Years</option>
      </select>
      <select id="venue-filter">
        <option value="">All Venues</option>
      </select>
      <select id="keyword-filter">
        <option value="">All Keywords</option>
      </select>
      <button id="reset-btn">Reset</button>
      <button id="stats-btn" class="stats-btn">📊 Statistics</button>
      <button id="add-arxiv-btn" class="add-arxiv-btn" title="Add paper from arXiv (requires local server)">➕ Add from arXiv</button>
      <button id="upload-github-btn" class="upload-github-btn" title="Upload changes to GitHub (requires local server)">☁️ Upload to GitHub</button>
    </div>
    <div class="search-stats">
      <span id="result-count">Loading...</span>
    </div>
  </div>

  <!-- Statistics Panel -->
  <div id="stats-panel" class="stats-panel" style="display: none;">
    <div class="stats-header">
      <h2>📊 Paper Statistics</h2>
      <button id="close-stats" class="close-stats">✕</button>
    </div>
    <div class="stats-content">
      <div class="stat-card">
        <h3>Year Distribution</h3>
        <div id="year-chart" class="bar-chart"></div>
      </div>
      <div class="stat-card">
        <h3>Top 10 Venues</h3>
        <div id="venue-chart" class="horizontal-bar-chart"></div>
      </div>
      <div class="stat-card">
        <h3>Top 10 Authors</h3>
        <div id="author-chart" class="horizontal-bar-chart"></div>
      </div>
      <div class="stat-card">
        <h3>Top 10 Institutions</h3>
        <div id="institution-chart" class="horizontal-bar-chart"></div>
      </div>
      <div class="stat-card">
        <h3>Top 20 Keywords</h3>
        <div id="keyword-cloud" class="keyword-cloud"></div>
      </div>
    </div>
  </div>

  <!-- Add from arXiv Modal -->
  <div id="arxiv-modal" class="arxiv-modal" style="display: none;">
    <div class="arxiv-modal-content">
      <div class="arxiv-modal-header">
        <h2>➕ Add Paper from arXiv</h2>
        <button id="close-arxiv" class="close-arxiv">✕</button>
      </div>
      <div class="arxiv-modal-body">
        <div class="form-group">
          <label for="arxiv-id-input">arXiv ID *</label>
          <input type="text" id="arxiv-id-input" placeholder="e.g., 2301.12345 or 2301.12345v1" required>
          <small>Enter the arXiv ID from the paper URL (e.g., arxiv.org/abs/2301.12345)</small>
        </div>
        <div class="form-group">
          <label for="abbr-input">Abbreviation (optional)</label>
          <input type="text" id="abbr-input" placeholder="e.g., FlashAttn2, GPT4">
          <small>Leave empty to auto-generate a random abbreviation</small>
        </div>
        <div id="arxiv-status" class="arxiv-status"></div>
      </div>
      <div class="arxiv-modal-footer">
        <button id="arxiv-submit-btn" class="btn-primary">Add Paper</button>
        <button id="arxiv-cancel-btn" class="btn-secondary">Cancel</button>
      </div>
    </div>
  </div>

  <!-- Upload to GitHub Modal -->
  <div id="github-modal" class="arxiv-modal" style="display: none;">
    <div class="arxiv-modal-content">
      <div class="arxiv-modal-header">
        <h2>☁️ Upload to GitHub</h2>
        <button id="close-github" class="close-arxiv">✕</button>
      </div>
      <div class="arxiv-modal-body">
        <div class="form-group">
          <label for="commit-message-input">Commit Message *</label>
          <input type="text" id="commit-message-input" placeholder="e.g., Add new paper or Update notes" required>
          <small>Describe the changes you made</small>
        </div>
        <div id="github-status" class="arxiv-status"></div>
      </div>
      <div class="arxiv-modal-footer">
        <button id="github-submit-btn" class="btn-primary">Upload</button>
        <button id="github-cancel-btn" class="btn-secondary">Cancel</button>
      </div>
    </div>
  </div>

  <div id="paper-list" class="paper-list">
    <!-- Papers will be rendered here -->
  </div>

  <!-- Lightbox for image preview -->
  <div id="lightbox" class="lightbox-overlay">
    <img id="lightbox-img" src="" alt="Preview">
    <span class="lightbox-hint">Press ESC or click to close</span>
  </div>
</div>

<style>
/* Search App Container */
#paper-search-app {
  max-width: 1200px;
  margin: 0 auto;
}

/* Search Header */
.search-header {
  position: sticky;
  top: 0;
  background: #fff;
  padding: 15px 0;
  z-index: 100;
  border-bottom: 1px solid #e0e0e0;
  margin-bottom: 20px;
}

.search-box {
  margin-bottom: 12px;
}

#search-input {
  width: 100%;
  padding: 12px 16px;
  font-size: 16px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
}

#search-input:focus {
  border-color: #4a90d9;
  box-shadow: 0 0 0 3px rgba(74, 144, 217, 0.15);
}

.search-hint {
  display: block;
  margin-top: 6px;
  font-size: 12px;
  color: #888;
}

.search-hint kbd {
  background: #f5f5f5;
  border: 1px solid #ddd;
  border-radius: 3px;
  padding: 1px 5px;
  font-family: monospace;
  font-size: 11px;
}

/* Filter Bar */
.filter-bar {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}

.filter-bar select {
  padding: 8px 12px;
  font-size: 14px;
  border: 1px solid #ddd;
  border-radius: 6px;
  background: #fff;
  cursor: pointer;
  min-width: 130px;
}

.filter-bar select:focus {
  outline: none;
  border-color: #4a90d9;
}

#reset-btn {
  padding: 8px 16px;
  font-size: 14px;
  border: 1px solid #ddd;
  border-radius: 6px;
  background: #f8f8f8;
  cursor: pointer;
  transition: background 0.2s;
}

#reset-btn:hover {
  background: #eee;
}

.stats-btn {
  padding: 8px 16px;
  font-size: 14px;
  border: 1px solid #4a90d9;
  border-radius: 6px;
  background: #4a90d9;
  color: white;
  cursor: pointer;
  transition: background 0.2s;
}

.stats-btn:hover {
  background: #3a7bc8;
}

.add-arxiv-btn {
  padding: 8px 16px;
  font-size: 14px;
  border: 1px solid #43a047;
  border-radius: 6px;
  background: #43a047;
  color: white;
  cursor: pointer;
  transition: background 0.2s;
}

.add-arxiv-btn:hover {
  background: #388e3c;
}

.upload-github-btn {
  padding: 8px 16px;
  font-size: 14px;
  border: 1px solid #f57c00;
  border-radius: 6px;
  background: #f57c00;
  color: white;
  cursor: pointer;
  transition: background 0.2s;
}

.upload-github-btn:hover {
  background: #e65100;
}

/* Statistics Panel */
.stats-panel {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  z-index: 1000;
  overflow-y: auto;
  animation: fadeIn 0.2s;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.stats-header {
  background: white;
  padding: 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 2px solid #e0e0e0;
  position: sticky;
  top: 0;
  z-index: 10;
}

.stats-header h2 {
  margin: 0;
  color: #2c3e50;
}

.close-stats {
  background: none;
  border: none;
  font-size: 28px;
  color: #888;
  cursor: pointer;
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: background 0.2s;
}

.close-stats:hover {
  background: #f0f0f0;
  color: #333;
}

.stats-content {
  background: white;
  max-width: 1200px;
  margin: 0 auto;
  padding: 30px;
  display: grid;
  grid-template-columns: 1fr;
  gap: 30px;
}

.stat-card {
  background: #f9f9f9;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 20px;
}

.stat-card h3 {
  margin: 0 0 15px 0;
  color: #2c3e50;
  font-size: 18px;
}

/* Bar Chart */
.bar-chart {
  display: flex;
  align-items: flex-end;
  gap: 4px;
  height: 200px;
  padding: 10px 0;
}

.bar-item {
  flex: 1;
  background: linear-gradient(to top, #4a90d9, #6aa5e3);
  border-radius: 4px 4px 0 0;
  position: relative;
  min-width: 20px;
  cursor: pointer;
  transition: opacity 0.2s;
}

.bar-item:hover {
  opacity: 0.8;
}

.bar-label {
  position: absolute;
  bottom: -25px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 11px;
  color: #666;
  white-space: nowrap;
}

.bar-value {
  position: absolute;
  top: -20px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 11px;
  font-weight: 600;
  color: #333;
}

/* Horizontal Bar Chart */
.horizontal-bar-chart {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.h-bar-item {
  display: flex;
  align-items: center;
  gap: 10px;
}

.h-bar-label {
  min-width: 100px;
  font-size: 13px;
  color: #666;
  text-align: right;
}

.h-bar-container {
  flex: 1;
  height: 24px;
  background: #e8e8e8;
  border-radius: 4px;
  overflow: hidden;
  position: relative;
}

.h-bar-fill {
  height: 100%;
  background: linear-gradient(to right, #7b1fa2, #9c27b0);
  border-radius: 4px;
  transition: width 0.5s ease;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 8px;
}

.h-bar-value {
  font-size: 11px;
  font-weight: 600;
  color: white;
}

/* Keyword Cloud */
.keyword-cloud {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 10px;
  justify-content: center;
}

.keyword-tag {
  padding: 6px 12px;
  border-radius: 16px;
  background: #e8f5e9;
  color: #2e7d32;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
  border: 1px solid #c8e6c9;
}

.keyword-tag:hover {
  transform: translateY(-2px);
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

.keyword-tag.size-xl {
  font-size: 18px;
  padding: 8px 16px;
}

.keyword-tag.size-lg {
  font-size: 16px;
  padding: 7px 14px;
}

.keyword-tag.size-md {
  font-size: 14px;
  padding: 6px 12px;
}

.keyword-tag.size-sm {
  font-size: 12px;
  padding: 5px 10px;
}

/* arXiv Modal */
.arxiv-modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  animation: fadeIn 0.2s;
}

.arxiv-modal-content {
  background: white;
  border-radius: 8px;
  width: 90%;
  max-width: 500px;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.arxiv-modal-header {
  padding: 20px;
  border-bottom: 2px solid #e0e0e0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.arxiv-modal-header h2 {
  margin: 0;
  color: #2c3e50;
  font-size: 20px;
}

.close-arxiv {
  background: none;
  border: none;
  font-size: 28px;
  color: #888;
  cursor: pointer;
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: background 0.2s;
}

.close-arxiv:hover {
  background: #f0f0f0;
  color: #333;
}

.arxiv-modal-body {
  padding: 20px;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 6px;
  font-weight: 600;
  color: #333;
  font-size: 14px;
}

.form-group input {
  width: 100%;
  padding: 10px 12px;
  font-size: 14px;
  border: 2px solid #e0e0e0;
  border-radius: 6px;
  outline: none;
  transition: border-color 0.2s;
  box-sizing: border-box;
}

.form-group input:focus {
  border-color: #4a90d9;
}

.form-group small {
  display: block;
  margin-top: 4px;
  font-size: 12px;
  color: #888;
}

.arxiv-status {
  min-height: 20px;
  padding: 10px;
  border-radius: 4px;
  font-size: 13px;
  margin-top: 10px;
}

.arxiv-status.status-loading {
  background: #e3f2fd;
  color: #1565c0;
}

.arxiv-status.status-success {
  background: #e8f5e9;
  color: #2e7d32;
}

.arxiv-status.status-error {
  background: #ffebee;
  color: #c62828;
}

.arxiv-modal-footer {
  padding: 15px 20px;
  border-top: 1px solid #e0e0e0;
  display: flex;
  gap: 10px;
  justify-content: flex-end;
}

.btn-primary {
  padding: 10px 20px;
  font-size: 14px;
  border: none;
  border-radius: 6px;
  background: #43a047;
  color: white;
  cursor: pointer;
  font-weight: 500;
  transition: background 0.2s;
}

.btn-primary:hover {
  background: #388e3c;
}

.btn-primary:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.btn-secondary {
  padding: 10px 20px;
  font-size: 14px;
  border: 1px solid #ddd;
  border-radius: 6px;
  background: #f8f8f8;
  color: #333;
  cursor: pointer;
  font-weight: 500;
  transition: background 0.2s;
}

.btn-secondary:hover {
  background: #eee;
}

/* Search Stats */
.search-stats {
  font-size: 14px;
  color: #666;
}

#result-count {
  font-weight: 500;
}

/* Paper List */
.paper-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* Paper Card */
.paper-card {
  display: flex;
  gap: 16px;
  padding: 16px 20px;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  background: #fff;
  transition: box-shadow 0.2s, border-color 0.2s;
}

.paper-card:hover {
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  border-color: #d0d0d0;
}

.paper-cover {
  flex-shrink: 0;
  width: 160px;
  height: 100px;
  border-radius: 6px;
  overflow: hidden;
  background: #f5f5f5;
  display: flex;
  align-items: center;
  justify-content: center;
}

.paper-cover img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  cursor: pointer;
  transition: transform 0.2s;
}

.paper-cover img:hover {
  transform: scale(1.05);
}

.paper-cover .no-cover {
  color: #ccc;
  font-size: 12px;
}

.paper-content {
  flex: 1;
  min-width: 0;
}

.paper-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 8px;
  line-height: 1.4;
}

.paper-title a {
  color: #2c3e50;
  text-decoration: none;
}

.paper-title a:hover {
  color: #4a90d9;
}

.paper-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
  font-size: 13px;
}

.paper-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
}

.badge-year {
  background: #e3f2fd;
  color: #1565c0;
}

.badge-venue {
  background: #f3e5f5;
  color: #7b1fa2;
}

.badge-keyword {
  background: #e8f5e9;
  color: #2e7d32;
}

.paper-authors {
  font-size: 13px;
  color: #666;
  margin-bottom: 6px;
}

.paper-institutions {
  font-size: 12px;
  color: #888;
}

.paper-links {
  margin-top: 10px;
  display: flex;
  gap: 12px;
}

.paper-links a {
  font-size: 13px;
  color: #4a90d9;
  text-decoration: none;
}

.paper-links a:hover {
  text-decoration: underline;
}

/* Highlight matched text */
.highlight {
  background: #fff3cd;
  padding: 0 2px;
  border-radius: 2px;
}

/* No results */
.no-results {
  text-align: center;
  padding: 40px;
  color: #888;
}

/* Lightbox */
.lightbox-overlay {
  display: flex;
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0);
  z-index: 1000;
  justify-content: center;
  align-items: center;
  cursor: zoom-out;
  pointer-events: none;
  visibility: hidden;
  transition: background 0.3s ease, visibility 0s linear 0.3s;
}

.lightbox-overlay.active {
  background: rgba(0, 0, 0, 0.9);
  pointer-events: auto;
  visibility: visible;
  transition: background 0.3s ease, visibility 0s linear 0s;
}

.lightbox-overlay img {
  max-width: 90%;
  max-height: 90%;
  object-fit: contain;
  border-radius: 4px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
  transform: scale(0.3);
  opacity: 0;
  transition: transform 0.3s ease, opacity 0.3s ease;
}

.lightbox-overlay.active img {
  transform: scale(1);
  opacity: 1;
}

.lightbox-hint {
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  color: #fff;
  font-size: 14px;
  opacity: 0;
  transition: opacity 0.3s ease 0.1s;
}

.lightbox-overlay.active .lightbox-hint {
  opacity: 0.7;
}

/* Loading */
.loading {
  text-align: center;
  padding: 40px;
  color: #888;
}

/* Pagination Controls */
.pagination-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 24px;
  padding: 16px 20px;
  background: #f9f9f9;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  flex-wrap: wrap;
  gap: 12px;
}

.pagination-info {
  font-size: 14px;
  color: #666;
  font-weight: 500;
}

.pagination-buttons {
  display: flex;
  gap: 6px;
  align-items: center;
  flex-wrap: wrap;
}

.pagination-btn {
  padding: 6px 12px;
  font-size: 14px;
  border: 1px solid #ddd;
  border-radius: 6px;
  background: #fff;
  color: #4a90d9;
  cursor: pointer;
  transition: all 0.2s;
  min-width: 40px;
  font-weight: 500;
}

.pagination-btn:hover:not(:disabled) {
  background: #4a90d9;
  color: white;
  border-color: #4a90d9;
}

.pagination-btn.active {
  background: #4a90d9;
  color: white;
  border-color: #4a90d9;
}

.pagination-btn:disabled {
  background: #f5f5f5;
  color: #ccc;
  border-color: #e0e0e0;
  cursor: not-allowed;
}

.pagination-ellipsis {
  padding: 6px 4px;
  color: #888;
  font-size: 14px;
}

/* Responsive */
@media (max-width: 768px) {
  .paper-card {
    flex-direction: column;
  }

  .paper-cover {
    width: 100%;
    height: 150px;
  }
}

@media (max-width: 600px) {
  .filter-bar {
    flex-direction: column;
  }

  .filter-bar select {
    width: 100%;
  }

  .paper-meta {
    flex-direction: column;
    gap: 4px;
  }
}
</style>

<script>
// Lightbox functions (global scope)
function openLightbox(event, src) {
  event.preventDefault();
  event.stopPropagation();
  const lightbox = document.getElementById('lightbox');
  const lightboxImg = document.getElementById('lightbox-img');
  lightboxImg.src = src;
  lightbox.classList.add('active');
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  const lightbox = document.getElementById('lightbox');
  lightbox.classList.remove('active');
  document.body.style.overflow = '';
}

(function() {
  let papers = [];
  let filteredPapers = [];
  let currentPage = 1;
  const itemsPerPage = 10;

  // Load paper data
  async function loadPapers() {
    try {
      // 使用基于站点根目录的绝对路径
      const basePath = window.location.pathname.includes('/search') ? '..' : '.';
      const response = await fetch(basePath + '/js/papers.json');
      const data = await response.json();
      papers = data.papers;

      // Populate filter options
      populateFilters(data.filters);

      // Initial render
      filteredPapers = papers;
      renderPapers(filteredPapers);
      updateStats();
    } catch (error) {
      console.error('Failed to load papers:', error);
      document.getElementById('paper-list').innerHTML =
        '<div class="no-results">Failed to load paper data</div>';
    }
  }

  // Populate filter dropdowns
  function populateFilters(filters) {
    const yearSelect = document.getElementById('year-filter');
    const venueSelect = document.getElementById('venue-filter');
    const keywordSelect = document.getElementById('keyword-filter');

    filters.years.forEach(year => {
      yearSelect.innerHTML += `<option value="${year}">${year}</option>`;
    });

    filters.venues.forEach(venue => {
      venueSelect.innerHTML += `<option value="${venue}">${venue}</option>`;
    });

    filters.keywords.forEach(keyword => {
      keywordSelect.innerHTML += `<option value="${keyword}">${keyword}</option>`;
    });
  }

  // Filter papers
  function filterPapers() {
    const searchQuery = document.getElementById('search-input').value.toLowerCase().trim();
    const yearFilter = document.getElementById('year-filter').value;
    const venueFilter = document.getElementById('venue-filter').value;
    const keywordFilter = document.getElementById('keyword-filter').value;

    filteredPapers = papers.filter(paper => {
      // Search query
      if (searchQuery) {
        const searchFields = [
          paper.title,
          paper.abbr,
          ...paper.authors,
          ...paper.institutions
        ].join(' ').toLowerCase();

        if (!searchFields.includes(searchQuery)) {
          return false;
        }
      }

      // Year filter
      if (yearFilter && paper.year != yearFilter) {
        return false;
      }

      // Venue filter
      if (venueFilter && paper.venue !== venueFilter) {
        return false;
      }

      // Keyword filter
      if (keywordFilter && !paper.keywords.includes(keywordFilter)) {
        return false;
      }

      return true;
    });

    // Reset to page 1 when filtering
    currentPage = 1;
    renderPapers(filteredPapers);
    updateStats();
  }

  // Highlight matched text
  function highlightText(text, query) {
    if (!query || !text) return text;
    const regex = new RegExp(`(${escapeRegex(query)})`, 'gi');
    return text.replace(regex, '<span class="highlight">$1</span>');
  }

  function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  // Render papers
  function renderPapers(papersToRender) {
    const container = document.getElementById('paper-list');
    const searchQuery = document.getElementById('search-input').value.toLowerCase().trim();

    if (papersToRender.length === 0) {
      container.innerHTML = '<div class="no-results">No papers found matching your criteria</div>';
      renderPagination(0);
      return;
    }

    // Calculate pagination
    const totalPages = Math.ceil(papersToRender.length / itemsPerPage);
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const papersToShow = papersToRender.slice(startIndex, endIndex);

    const html = papersToShow.map(paper => {
      const title = highlightText(paper.title, searchQuery);
      const authors = paper.authors.map(a => highlightText(a, searchQuery)).join(', ');
      const institutions = paper.institutions.map(i => highlightText(i, searchQuery)).join(', ');

      // 封面图片
      const coverHtml = paper.cover
        ? `<img src="${paper.cover}" alt="cover" loading="lazy" onclick="openLightbox(event, this.src)">`
        : `<span class="no-cover">No Cover</span>`;

      return `
        <div class="paper-card">
          <div class="paper-cover">
            ${coverHtml}
          </div>
          <div class="paper-content">
            <div class="paper-title">
              ${paper.url ? `<a href="${paper.url}" target="_blank">${title}</a>` : title}
              ${paper.abbr ? `<span style="color:#888;font-weight:normal;"> (${paper.abbr})</span>` : ''}
            </div>
            <div class="paper-meta">
              <span class="paper-badge badge-year">${paper.year}</span>
              <span class="paper-badge badge-venue">${paper.venue}</span>
              ${paper.keywords.map(k => `<span class="paper-badge badge-keyword">${k}</span>`).join('')}
            </div>
            ${authors ? `<div class="paper-authors">${authors}</div>` : ''}
            ${institutions ? `<div class="paper-institutions">${institutions}</div>` : ''}
            <div class="paper-links">
              ${paper.code_url ? `<a href="${paper.code_url}" target="_blank">Code</a>` : ''}
              ${paper.note_url ? `<a href="${paper.note_url}" target="_blank">Note</a>` : ''}
              ${paper.prototxt_path ? `<a href="../edit.html?path=${encodeURIComponent(paper.prototxt_path)}" target="_blank" title="Requires local server">Edit 🔧</a>` : ''}
            </div>
          </div>
        </div>
      `;
    }).join('');

    container.innerHTML = html;
    renderPagination(totalPages);
  }

  // Update stats
  function updateStats() {
    const total = papers.length;
    const showing = filteredPapers.length;
    const countEl = document.getElementById('result-count');

    if (showing === total) {
      countEl.textContent = `${total} papers`;
    } else {
      countEl.textContent = `${showing} of ${total} papers`;
    }
  }

  // Render pagination controls
  function renderPagination(totalPages) {
    const container = document.getElementById('paper-list');

    // Remove existing pagination if it exists
    const existingPagination = document.getElementById('pagination-controls');
    if (existingPagination) {
      existingPagination.remove();
    }

    if (totalPages <= 1) {
      return;
    }

    const startIndex = (currentPage - 1) * itemsPerPage + 1;
    const endIndex = Math.min(currentPage * itemsPerPage, filteredPapers.length);

    let paginationHtml = `
      <div id="pagination-controls" class="pagination-controls">
        <div class="pagination-info">
          Showing ${startIndex}-${endIndex} of ${filteredPapers.length} papers
        </div>
        <div class="pagination-buttons">
    `;

    // Previous button
    paginationHtml += `
      <button class="pagination-btn" ${currentPage === 1 ? 'disabled' : ''} data-page="${currentPage - 1}">
        ← Previous
      </button>
    `;

    // Page numbers
    const maxVisiblePages = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);

    if (endPage - startPage < maxVisiblePages - 1) {
      startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }

    if (startPage > 1) {
      paginationHtml += `<button class="pagination-btn" data-page="1">1</button>`;
      if (startPage > 2) {
        paginationHtml += `<span class="pagination-ellipsis">...</span>`;
      }
    }

    for (let i = startPage; i <= endPage; i++) {
      const isActive = i === currentPage ? 'active' : '';
      paginationHtml += `<button class="pagination-btn ${isActive}" data-page="${i}">${i}</button>`;
    }

    if (endPage < totalPages) {
      if (endPage < totalPages - 1) {
        paginationHtml += `<span class="pagination-ellipsis">...</span>`;
      }
      paginationHtml += `<button class="pagination-btn" data-page="${totalPages}">${totalPages}</button>`;
    }

    // Next button
    paginationHtml += `
      <button class="pagination-btn" ${currentPage === totalPages ? 'disabled' : ''} data-page="${currentPage + 1}">
        Next →
      </button>
    `;

    paginationHtml += `
        </div>
      </div>
    `;

    container.insertAdjacentHTML('afterend', paginationHtml);

    // Add event listeners to pagination buttons
    document.querySelectorAll('.pagination-btn').forEach(btn => {
      btn.addEventListener('click', function() {
        if (this.disabled) return;
        const page = parseInt(this.dataset.page);
        goToPage(page);
      });
    });
  }

  // Go to specific page
  function goToPage(page) {
    currentPage = page;
    renderPapers(filteredPapers);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // Reset filters
  function resetFilters() {
    document.getElementById('search-input').value = '';
    document.getElementById('year-filter').value = '';
    document.getElementById('venue-filter').value = '';
    document.getElementById('keyword-filter').value = '';
    filterPapers();
  }

  // Generate and show statistics
  function showStatistics() {
    const panel = document.getElementById('stats-panel');
    panel.style.display = 'block';
    document.body.style.overflow = 'hidden';

    generateYearChart();
    generateVenueChart();
    generateAuthorChart();
    generateInstitutionChart();
    generateKeywordCloud();
  }

  function closeStatistics() {
    document.getElementById('stats-panel').style.display = 'none';
    document.body.style.overflow = '';
  }

  function generateYearChart() {
    const yearCounts = {};
    papers.forEach(paper => {
      yearCounts[paper.year] = (yearCounts[paper.year] || 0) + 1;
    });

    const sortedYears = Object.keys(yearCounts).sort((a, b) => a - b);
    const maxCount = Math.max(...Object.values(yearCounts));

    const chartHtml = sortedYears.map(year => {
      const count = yearCounts[year];
      const height = (count / maxCount) * 100;
      return `
        <div class="bar-item" style="height: ${height}%">
          <span class="bar-value">${count}</span>
          <span class="bar-label">${year}</span>
        </div>
      `;
    }).join('');

    document.getElementById('year-chart').innerHTML = chartHtml;
  }

  function generateVenueChart() {
    const venueCounts = {};
    papers.forEach(paper => {
      venueCounts[paper.venue] = (venueCounts[paper.venue] || 0) + 1;
    });

    const sortedVenues = Object.entries(venueCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);

    const maxCount = sortedVenues[0][1];

    const chartHtml = sortedVenues.map(([venue, count]) => {
      const width = (count / maxCount) * 100;
      return `
        <div class="h-bar-item">
          <div class="h-bar-label">${venue}</div>
          <div class="h-bar-container">
            <div class="h-bar-fill" style="width: ${width}%">
              <span class="h-bar-value">${count}</span>
            </div>
          </div>
        </div>
      `;
    }).join('');

    document.getElementById('venue-chart').innerHTML = chartHtml;
  }

  function generateAuthorChart() {
    const authorCounts = {};
    papers.forEach(paper => {
      paper.authors.forEach(author => {
        authorCounts[author] = (authorCounts[author] || 0) + 1;
      });
    });

    const sortedAuthors = Object.entries(authorCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);

    if (sortedAuthors.length === 0) {
      document.getElementById('author-chart').innerHTML = '<div class="no-data">No author data available</div>';
      return;
    }

    const maxCount = sortedAuthors[0][1];

    const chartHtml = sortedAuthors.map(([author, count]) => {
      const width = (count / maxCount) * 100;
      return `
        <div class="h-bar-item">
          <div class="h-bar-label">${author}</div>
          <div class="h-bar-container">
            <div class="h-bar-fill" style="width: ${width}%">
              <span class="h-bar-value">${count}</span>
            </div>
          </div>
        </div>
      `;
    }).join('');

    document.getElementById('author-chart').innerHTML = chartHtml;
  }

  function generateInstitutionChart() {
    const institutionCounts = {};
    papers.forEach(paper => {
      paper.institutions.forEach(institution => {
        institutionCounts[institution] = (institutionCounts[institution] || 0) + 1;
      });
    });

    const sortedInstitutions = Object.entries(institutionCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);

    if (sortedInstitutions.length === 0) {
      document.getElementById('institution-chart').innerHTML = '<div class="no-data">No institution data available</div>';
      return;
    }

    const maxCount = sortedInstitutions[0][1];

    const chartHtml = sortedInstitutions.map(([institution, count]) => {
      const width = (count / maxCount) * 100;
      return `
        <div class="h-bar-item">
          <div class="h-bar-label">${institution}</div>
          <div class="h-bar-container">
            <div class="h-bar-fill" style="width: ${width}%">
              <span class="h-bar-value">${count}</span>
            </div>
          </div>
        </div>
      `;
    }).join('');

    document.getElementById('institution-chart').innerHTML = chartHtml;
  }

  function generateKeywordCloud() {
    const keywordCounts = {};
    papers.forEach(paper => {
      paper.keywords.forEach(keyword => {
        keywordCounts[keyword] = (keywordCounts[keyword] || 0) + 1;
      });
    });

    const sortedKeywords = Object.entries(keywordCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20);

    const maxCount = sortedKeywords[0][1];
    const minCount = sortedKeywords[sortedKeywords.length - 1][1];

    const cloudHtml = sortedKeywords.map(([keyword, count]) => {
      let sizeClass = 'size-sm';
      const ratio = (count - minCount) / (maxCount - minCount);

      if (ratio > 0.75) sizeClass = 'size-xl';
      else if (ratio > 0.5) sizeClass = 'size-lg';
      else if (ratio > 0.25) sizeClass = 'size-md';

      return `<span class="keyword-tag ${sizeClass}" title="${count} papers">${keyword}</span>`;
    }).join('');

    document.getElementById('keyword-cloud').innerHTML = cloudHtml;
  }

  // arXiv Modal functions
  function showArxivModal() {
    const modal = document.getElementById('arxiv-modal');
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    document.getElementById('arxiv-id-input').focus();
    document.getElementById('arxiv-status').textContent = '';
    document.getElementById('arxiv-status').className = 'arxiv-status';
  }

  function closeArxivModal() {
    const modal = document.getElementById('arxiv-modal');
    modal.style.display = 'none';
    document.body.style.overflow = '';
    document.getElementById('arxiv-id-input').value = '';
    document.getElementById('abbr-input').value = '';
    document.getElementById('arxiv-status').textContent = '';
    document.getElementById('arxiv-status').className = 'arxiv-status';
  }

  async function addPaperFromArxiv() {
    const arxivId = document.getElementById('arxiv-id-input').value.trim();
    const abbr = document.getElementById('abbr-input').value.trim();
    const statusEl = document.getElementById('arxiv-status');
    const submitBtn = document.getElementById('arxiv-submit-btn');

    if (!arxivId) {
      statusEl.textContent = 'Please enter an arXiv ID';
      statusEl.className = 'arxiv-status status-error';
      return;
    }

    // Show loading state
    statusEl.textContent = 'Adding paper from arXiv... This may take a few moments.';
    statusEl.className = 'arxiv-status status-loading';
    submitBtn.disabled = true;

    try {
      const response = await fetch('http://localhost:8001/api/add-from-arxiv', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          arxiv_id: arxivId,
          abbr: abbr
        })
      });

      const result = await response.json();

      if (response.ok && result.success) {
        statusEl.textContent = `✓ Paper added successfully! Abbreviation: ${result.abbr}. Auto-reload is running. Refresh your browser in a few seconds to see the new paper.`;
        statusEl.className = 'arxiv-status status-success';

        // Clear inputs after success
        setTimeout(() => {
          closeArxivModal();
        }, 3000);
      } else {
        throw new Error(result.error || 'Failed to add paper');
      }
    } catch (error) {
      statusEl.textContent = `✗ Error: ${error.message}. Make sure the server is running and the arXiv ID is valid.`;
      statusEl.className = 'arxiv-status status-error';
      submitBtn.disabled = false;
    }
  }

  // GitHub Upload Modal functions
  function showGithubModal() {
    const modal = document.getElementById('github-modal');
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    document.getElementById('commit-message-input').focus();
    document.getElementById('github-status').textContent = '';
    document.getElementById('github-status').className = 'arxiv-status';
  }

  function closeGithubModal() {
    const modal = document.getElementById('github-modal');
    modal.style.display = 'none';
    document.body.style.overflow = '';
    document.getElementById('commit-message-input').value = '';
    document.getElementById('github-status').textContent = '';
    document.getElementById('github-status').className = 'arxiv-status';
  }

  async function uploadToGithub() {
    const commitMessage = document.getElementById('commit-message-input').value.trim();
    const statusEl = document.getElementById('github-status');
    const submitBtn = document.getElementById('github-submit-btn');

    if (!commitMessage) {
      statusEl.textContent = 'Please enter a commit message';
      statusEl.className = 'arxiv-status status-error';
      return;
    }

    // Show loading state
    statusEl.textContent = 'Uploading to GitHub... This may take a few moments.';
    statusEl.className = 'arxiv-status status-loading';
    submitBtn.disabled = true;

    try {
      // First check if server is reachable
      console.log('[GitHub Upload] Checking server connectivity...');
      statusEl.textContent = 'Checking server connection...';

      let serverCheck;
      try {
        serverCheck = await fetch('http://localhost:8001/api/get-keywords', { method: 'GET' });
        console.log('[GitHub Upload] Server check response:', serverCheck.status);
        if (!serverCheck.ok) {
          throw new Error(`Server responded with status ${serverCheck.status}`);
        }
      } catch (e) {
        console.error('[GitHub Upload] Server check failed:', e);
        throw new Error('Cannot connect to server at localhost:8001. Please make sure ./start_editor.sh is running.');
      }

      console.log('[GitHub Upload] Server is reachable, sending upload request...');
      statusEl.textContent = 'Uploading to GitHub... This may take a few moments.';

      const response = await fetch('http://localhost:8001/api/upload-github', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          commit_message: commitMessage
        })
      });

      console.log('[GitHub Upload] Upload response status:', response.status);
      const result = await response.json();
      console.log('[GitHub Upload] Upload result:', result);

      if (response.ok && result.success) {
        statusEl.textContent = '✓ Successfully uploaded to GitHub and deployed to GitHub Pages!';
        statusEl.className = 'arxiv-status status-success';

        // Clear input after success
        setTimeout(() => {
          closeGithubModal();
        }, 3000);
      } else {
        throw new Error(result.error || 'Failed to upload');
      }
    } catch (error) {
      console.error('[GitHub Upload] Error:', error);
      statusEl.textContent = `✗ Error: ${error.message}`;
      statusEl.className = 'arxiv-status status-error';
      submitBtn.disabled = false;
    }
  }

  // Debounce function
  function debounce(func, wait) {
    let timeout;
    return function(...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  }

  // Initialize
  function init() {
    loadPapers();

    // Event listeners
    const debouncedFilter = debounce(filterPapers, 200);

    document.getElementById('search-input').addEventListener('input', debouncedFilter);
    document.getElementById('year-filter').addEventListener('change', filterPapers);
    document.getElementById('venue-filter').addEventListener('change', filterPapers);
    document.getElementById('keyword-filter').addEventListener('change', filterPapers);
    document.getElementById('reset-btn').addEventListener('click', resetFilters);

    // Statistics panel
    document.getElementById('stats-btn').addEventListener('click', showStatistics);
    document.getElementById('close-stats').addEventListener('click', closeStatistics);

    // arXiv modal
    document.getElementById('add-arxiv-btn').addEventListener('click', showArxivModal);
    document.getElementById('close-arxiv').addEventListener('click', closeArxivModal);
    document.getElementById('arxiv-cancel-btn').addEventListener('click', closeArxivModal);
    document.getElementById('arxiv-submit-btn').addEventListener('click', addPaperFromArxiv);

    // Close arXiv modal on overlay click
    document.getElementById('arxiv-modal').addEventListener('click', (e) => {
      if (e.target.id === 'arxiv-modal') {
        closeArxivModal();
      }
    });

    // Submit on Enter in arXiv inputs
    document.getElementById('arxiv-id-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        addPaperFromArxiv();
      }
    });
    document.getElementById('abbr-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        addPaperFromArxiv();
      }
    });

    // GitHub upload modal
    document.getElementById('upload-github-btn').addEventListener('click', showGithubModal);
    document.getElementById('close-github').addEventListener('click', closeGithubModal);
    document.getElementById('github-cancel-btn').addEventListener('click', closeGithubModal);
    document.getElementById('github-submit-btn').addEventListener('click', uploadToGithub);

    // Close GitHub modal on overlay click
    document.getElementById('github-modal').addEventListener('click', (e) => {
      if (e.target.id === 'github-modal') {
        closeGithubModal();
      }
    });

    // Submit on Enter in commit message input
    document.getElementById('commit-message-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        uploadToGithub();
      }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {
        e.preventDefault();
        document.getElementById('search-input').focus();
      }
      if (e.key === 'Escape') {
        // Close GitHub modal first if open
        const githubModal = document.getElementById('github-modal');
        if (githubModal.style.display === 'flex') {
          closeGithubModal();
          return;
        }
        // Close arXiv modal if open
        const arxivModal = document.getElementById('arxiv-modal');
        if (arxivModal.style.display === 'flex') {
          closeArxivModal();
          return;
        }
        // Close stats panel if open
        const statsPanel = document.getElementById('stats-panel');
        if (statsPanel.style.display === 'block') {
          closeStatistics();
          return;
        }
        // Close lightbox if open
        const lightbox = document.getElementById('lightbox');
        if (lightbox.classList.contains('active')) {
          closeLightbox();
          return;
        }
        // Otherwise clear search input
        const input = document.getElementById('search-input');
        if (document.activeElement === input) {
          input.value = '';
          filterPapers();
          input.blur();
        }
      }
    });

    // Click overlay to close lightbox
    document.getElementById('lightbox').addEventListener('click', closeLightbox);
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
</script>
