// Auto-reload script for development
// Checks for content updates and reloads the page

(function() {
    // Configuration
    const CHECK_INTERVAL = 1000; // Check every second
    const RELOAD_DELAY = 500;    // Wait 500ms after detecting change

    // Files to monitor (relative to site root)
    const WATCH_FILES = [
        'js/papers.json',
        window.location.pathname
    ];

    let lastModified = {};
    let currentState = {
        scrollX: 0,
        scrollY: 0,
        searchParams: new URLSearchParams(window.location.search),
        hash: window.location.hash
    };

    // Save current state before reload
    function saveState() {
        currentState.scrollX = window.scrollX;
        currentState.scrollY = window.scrollY;
        currentState.searchParams = new URLSearchParams(window.location.search);
        currentState.hash = window.location.hash;

        // Save filter and sort states
        const sortFilter = document.getElementById('sort-filter');
        const yearFilter = document.getElementById('year-filter');
        const venueFilter = document.getElementById('venue-filter');
        const keywordFilter = document.getElementById('keyword-filter');
        const searchInput = document.getElementById('search-input');

        // Get current page number from the page info element or window variable
        let currentPageNum = 1;
        if (typeof window.currentPage !== 'undefined') {
            currentPageNum = window.currentPage;
        } else {
            // Try to get from page info text (e.g., "Page 2 of 5")
            const pageInfo = document.querySelector('.page-info');
            if (pageInfo) {
                const match = pageInfo.textContent.match(/Page\s+(\d+)/i);
                if (match) {
                    currentPageNum = parseInt(match[1], 10);
                }
            }
        }

        sessionStorage.setItem('auto-reload-state', JSON.stringify({
            scrollX: currentState.scrollX,
            scrollY: currentState.scrollY,
            searchParams: Array.from(currentState.searchParams.entries()),
            hash: currentState.hash,
            // Save UI filter states
            sortFilter: sortFilter ? sortFilter.value : '',
            yearFilter: yearFilter ? yearFilter.value : '',
            venueFilter: venueFilter ? venueFilter.value : '',
            keywordFilter: keywordFilter ? keywordFilter.value : '',
            searchInput: searchInput ? searchInput.value : '',
            // Save current page number
            currentPage: currentPageNum
        }));
    }

    // Restore state after reload
    function restoreState() {
        const saved = sessionStorage.getItem('auto-reload-state');
        if (!saved) return false;

        try {
            const state = JSON.parse(saved);
            sessionStorage.removeItem('auto-reload-state');

            console.log('[Auto-reload] Found saved state:', state);

            // Restore UI filter states
            // Note: Dynamic filters (year, venue, keyword) must be restored AFTER main page loads their options
            const restoreUI = () => {
                let hasChanges = false;

                // Restore sort filter immediately (options are static)
                if (state.sortFilter !== undefined) {
                    const sortFilter = document.getElementById('sort-filter');
                    if (sortFilter && sortFilter.value !== state.sortFilter) {
                        sortFilter.value = state.sortFilter;
                        hasChanges = true;
                        console.log('[Auto-reload] Restored sort filter:', state.sortFilter);
                    }
                }

                // Restore search input immediately
                if (state.searchInput !== undefined) {
                    const searchInput = document.getElementById('search-input');
                    if (searchInput && searchInput.value !== state.searchInput) {
                        searchInput.value = state.searchInput;
                        hasChanges = true;
                        console.log('[Auto-reload] Restored search input:', state.searchInput);
                    }
                }

                // Check if we have dynamic filters to restore
                const hasDynamicFilters = state.yearFilter || state.venueFilter || state.keywordFilter;

                // Trigger filter update if there were changes or we need to restore page/dynamic filters
                if (hasChanges || state.currentPage > 1 || hasDynamicFilters) {
                    // Function to dispatch events to all filters
                    const dispatchFilterEvents = () => {
                        const sortFilter = document.getElementById('sort-filter');
                        const yearFilter = document.getElementById('year-filter');
                        const venueFilter = document.getElementById('venue-filter');
                        const keywordFilter = document.getElementById('keyword-filter');
                        const searchInput = document.getElementById('search-input');

                        // Check if main page is initialized AND all elements exist
                        // window.papers is set by the main page's loadPapers() function
                        const mainPageReady = typeof window.papers !== 'undefined' && window.papers.length > 0 &&
                                              sortFilter && yearFilter && venueFilter && keywordFilter && searchInput;

                        // Also check if dynamic filter options are populated
                        const optionsPopulated = yearFilter && yearFilter.options.length > 1;

                        if (mainPageReady && optionsPopulated) {
                            console.log('[Auto-reload] Main page initialized, restoring dynamic filters');

                            // Now restore dynamic filters (year, venue, keyword) after options are loaded
                            if (state.yearFilter !== undefined && yearFilter.value !== state.yearFilter) {
                                yearFilter.value = state.yearFilter;
                                console.log('[Auto-reload] Restored year filter:', state.yearFilter);
                            }

                            if (state.venueFilter !== undefined && venueFilter.value !== state.venueFilter) {
                                venueFilter.value = state.venueFilter;
                                console.log('[Auto-reload] Restored venue filter:', state.venueFilter);
                            }

                            if (state.keywordFilter !== undefined && keywordFilter.value !== state.keywordFilter) {
                                keywordFilter.value = state.keywordFilter;
                                console.log('[Auto-reload] Restored keyword filter:', state.keywordFilter);
                            }

                            // Dispatch events to trigger filterPapers
                            // Each event will trigger filterPapers which reads all current values
                            sortFilter.dispatchEvent(new Event('change', { bubbles: true }));

                            // Restore page number after filters are applied
                            if (state.currentPage && state.currentPage > 1) {
                                setTimeout(() => {
                                    if (typeof window.goToPage === 'function') {
                                        console.log('[Auto-reload] Restoring page number:', state.currentPage);
                                        window.goToPage(state.currentPage);
                                    } else if (typeof window.currentPage !== 'undefined') {
                                        // Fallback: set the variable and trigger re-render
                                        window.currentPage = state.currentPage;
                                        if (typeof window.renderPapers === 'function' && typeof window.filteredPapers !== 'undefined') {
                                            window.renderPapers(window.filteredPapers);
                                        }
                                        console.log('[Auto-reload] Restored page via window.currentPage:', state.currentPage);
                                    }
                                }, 100);
                            }

                            console.log('[Auto-reload] All filters restored');
                            return true;
                        }
                        return false;
                    };

                    // Function to wait and retry dispatching events
                    const attemptDispatch = (retryCount = 0) => {
                        const maxRetries = 30; // 30 retries * 200ms = 6 seconds

                        if (dispatchFilterEvents()) {
                            // Success
                            return;
                        }

                        retryCount++;
                        if (retryCount < maxRetries) {
                            console.log(`[Auto-reload] Waiting for main page initialization... (${retryCount}/${maxRetries})`);
                            setTimeout(() => attemptDispatch(retryCount), 200);
                        } else {
                            console.log('[Auto-reload] Failed to dispatch events after max retries');
                            // Fallback: try anyway
                            const sortFilter = document.getElementById('sort-filter');
                            if (sortFilter) {
                                sortFilter.dispatchEvent(new Event('change', { bubbles: true }));
                            }
                        }
                    };

                    // Start after DOMContentLoaded and initial load
                    if (document.readyState === 'loading') {
                        // DOM still loading, wait for it
                        window.addEventListener('DOMContentLoaded', () => {
                            // Give main page time to initialize
                            setTimeout(() => attemptDispatch(0), 800);
                        });
                    } else {
                        // DOM already loaded
                        setTimeout(() => attemptDispatch(0), 800);
                    }
                }
            };

            // Restore scroll position with retry logic for dynamic content
            const restoreScroll = () => {
                const attemptRestore = (retryCount = 0) => {
                    const maxRetries = 20; // More retries for dynamic content

                    // Get current scroll position
                    const currentY = window.scrollY;
                    const currentX = window.scrollX;

                    // Check if we're already at the target position
                    if (currentY === state.scrollY && currentX === state.scrollX) {
                        console.log('[Auto-reload] Scroll position already correct');
                        return;
                    }

                    // Try to scroll
                    window.scrollTo(state.scrollX, state.scrollY);

                    // Log the attempt
                    console.log(`[Auto-reload] Attempt ${retryCount + 1}/${maxRetries}: scrolling to (${state.scrollX}, ${state.scrollY}), current: (${currentX}, ${currentY}), doc height: ${document.documentElement.scrollHeight}`);

                    // Check if scroll was successful
                    const newY = window.scrollY;
                    const newX = window.scrollX;

                    // If scroll didn't work or we're not at the target yet, retry
                    if ((newY !== state.scrollY || newX !== state.scrollX) && retryCount < maxRetries) {
                        setTimeout(() => attemptRestore(retryCount + 1), 150);
                    } else {
                        console.log('[Auto-reload] Successfully restored scroll position:', state.scrollX, state.scrollY);
                    }
                };

                // Start attempting after a short delay to allow initial render
                setTimeout(() => attemptRestore(0), 200);
            };

            // Restore UI immediately (doesn't depend on page load)
            restoreUI();

            // Wait for page to be ready before restoring scroll
            if (document.readyState === 'complete') {
                // Page already loaded, restore immediately
                restoreScroll();
            } else if (document.readyState === 'interactive') {
                // DOM ready but resources still loading, wait a bit
                window.addEventListener('load', restoreScroll);
            } else {
                // Not ready yet, wait for load
                window.addEventListener('load', restoreScroll);
            }

            return true;
        } catch (e) {
            console.error('[Auto-reload] Failed to restore state:', e);
            return false;
        }
    }

    // Get last modified time for a URL
    async function getLastModified(url) {
        try {
            const response = await fetch(url, { method: 'HEAD' });
            return {
                'last-modified': response.headers.get('last-modified'),
                'etag': response.headers.get('etag'),
                'content-length': response.headers.get('content-length')
            };
        } catch (e) {
            return null;
        }
    }

    // Initialize last modified times
    async function init() {
        for (const file of WATCH_FILES) {
            const info = await getLastModified(file);
            if (info) {
                lastModified[file] = info;
            }
        }
    }

    // Check for changes
    async function checkChanges() {
        for (const file of WATCH_FILES) {
            const info = await getLastModified(file);
            if (!info) continue;

            const previous = lastModified[file];
            if (!previous) {
                lastModified[file] = info;
                continue;
            }

            // Check if content changed (ETag or Last-Modified or Content-Length)
            const changed = info['etag'] && info['etag'] !== previous['etag'] ||
                           info['last-modified'] && info['last-modified'] !== previous['last-modified'] ||
                           info['content-length'] && info['content-length'] !== previous['content-length'];

            if (changed) {
                console.log(`[Auto-reload] Content changed: ${file}`);
                lastModified[file] = info;

                // Save current state before reload
                saveState();

                // Show notification
                showReloadNotification();

                // Reload after delay (use same URL which preserves search params and hash)
                setTimeout(() => {
                    console.log('[Auto-reload] Reloading page...');
                    location.reload();
                }, RELOAD_DELAY);
                return;
            }
        }
    }

    // Show reload notification
    function showReloadNotification() {
        const existing = document.getElementById('auto-reload-notification');
        if (existing) return;

        const notification = document.createElement('div');
        notification.id = 'auto-reload-notification';
        notification.textContent = 'Content updated, reloading...';
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #2196F3;
            color: white;
            padding: 12px 24px;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            z-index: 10000;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 14px;
            animation: slideIn 0.3s ease-out;
        `;

        // Add animation keyframes
        if (!document.getElementById('auto-reload-styles')) {
            const style = document.createElement('style');
            style.id = 'auto-reload-styles';
            style.textContent = `
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `;
            document.head.appendChild(style);
        }

        document.body.appendChild(notification);
    }

    // Only enable in development (localhost)
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        // Restore state if available - do this immediately
        const stateRestored = restoreState();

        // Start monitoring
        init().then(() => {
            console.log('[Auto-reload] Started monitoring for changes');
            setInterval(checkChanges, CHECK_INTERVAL);
        });
    }
})();
