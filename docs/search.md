# Paper Search

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
    </div>
    <div class="search-stats">
      <span id="result-count">Loading...</span>
    </div>
  </div>

  <div id="paper-list" class="paper-list">
    <!-- Papers will be rendered here -->
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

/* Loading */
.loading {
  text-align: center;
  padding: 40px;
  color: #888;
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
(function() {
  let papers = [];
  let filteredPapers = [];

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
      return;
    }

    const html = papersToRender.map(paper => {
      const title = highlightText(paper.title, searchQuery);
      const authors = paper.authors.map(a => highlightText(a, searchQuery)).join(', ');
      const institutions = paper.institutions.map(i => highlightText(i, searchQuery)).join(', ');

      // 封面图片
      const coverHtml = paper.cover
        ? `<img src="${paper.cover}" alt="cover" loading="lazy" onclick="window.open(this.src, '_blank')">`
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
            ${paper.code_url ? `
              <div class="paper-links">
                <a href="${paper.code_url}" target="_blank">Code</a>
              </div>
            ` : ''}
          </div>
        </div>
      `;
    }).join('');

    container.innerHTML = html;
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

  // Reset filters
  function resetFilters() {
    document.getElementById('search-input').value = '';
    document.getElementById('year-filter').value = '';
    document.getElementById('venue-filter').value = '';
    document.getElementById('keyword-filter').value = '';
    filterPapers();
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

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {
        e.preventDefault();
        document.getElementById('search-input').focus();
      }
      if (e.key === 'Escape') {
        const input = document.getElementById('search-input');
        if (document.activeElement === input) {
          input.value = '';
          filterPapers();
          input.blur();
        }
      }
    });
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
</script>
