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

                // Show notification
                showReloadNotification();

                // Reload after delay
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
        init().then(() => {
            console.log('[Auto-reload] Started monitoring for changes');
            setInterval(checkChanges, CHECK_INTERVAL);
        });
    }
})();
