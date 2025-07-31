// Markdown rendering function
// Enhanced Markdown to HTML converter with CSS Modules and Responsive Design
function renderMarkdown(markdownText) {
    // Split by newlines for paragraphs/sections
    const lines = markdownText.split('\n');
    let html = '<div class="article-container article-fade-in">';
    let inList = false;
    let inMeta = false;
    let metaLines = [];
    let beritaTerkait = false;
    let inBlockquote = false;
    let inCodeBlock = false;
    let codeBlockContent = '';
    let codeBlockLang = '';
    let tableOpen = false; // Track if table is open
    let inOrderedList = false;
    let listLevel = 0;
    let headerLevel = 0;

    // Helper function to escape HTML
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Helper function to process inline formatting
    function processInlineFormatting(text) {
        return text
            // Bold text
            .replace(/\*\*(.*?)\*\*/g, '<strong class="article-strong">$1</strong>')
            // Italic text
            .replace(/\*(.*?)\*/g, '<em class="article-emphasis">$1</em>')
            // Inline code
            .replace(/`(.*?)`/g, '<code class="article-inline-code">$1</code>')
            // Links
            .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="article-link">$1</a>')
            // Highlight important terms
            .replace(/\b(IMPORTANT|PENTING|KRITIS|URGENT)\b/gi, '<span class="article-highlight">$1</span>')
            // Highlight numbers and percentages
            .replace(/(\d+(?:\.\d+)?%?)/g, '<span class="article-number">$1</span>')
            // Highlight currency
            .replace(/(Rp\.?\s*\d+(?:,\d{3})*(?:\.\d{2})?)/gi, '<span class="article-currency">$1</span>');
    }

    lines.forEach((line, idx) => {
        const trimmed = line.trim();
        const escapedLine = escapeHtml(trimmed);

        // Markdown Headers (H1-H6)
        const headerMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
        if (headerMatch) {
            const level = headerMatch[1].length;
            const text = headerMatch[2];
            const headerClass = `article-h${level}`;
            html += `<h${level} class="${headerClass}">${processInlineFormatting(escapeHtml(text))}</h${level}>`;
            return;
        }

        // Headline (first non-empty line if no markdown headers)
        if (idx === 0 && trimmed.length > 0 && !trimmed.startsWith('#')) {
            html += `<h1 class="article-headline">${processInlineFormatting(escapedLine)}</h1>`;
            return;
        }

        // Byline (Ditulis oleh ...)
        if (/^Ditulis oleh/i.test(trimmed)) {
            html += `<div class="article-byline">
                <i class="fas fa-user-edit article-byline-icon"></i>
                <span>${processInlineFormatting(escapedLine)}</span>
            </div>`;
            inMeta = true;
            metaLines = [];
            return;
        }

        // Date or A+/A- (meta info)
        if (inMeta && (/^\d{1,2}\s+\w+\s+\d{4}$/.test(trimmed) || trimmed === 'A+' || trimmed === 'A−' || trimmed === 'A-')) {
            metaLines.push(trimmed);
            return;
        }
        if (inMeta && trimmed === '') {
            // End of meta block
            if (metaLines.length > 0) {
                html += `<div class="article-meta">`;
                metaLines.forEach((m, i) => {
                    if (i > 0) html += '<span class="mx-1">•</span>';
                    html += `<span class="article-meta-item">${escapeHtml(m)}</span>`;
                });
                html += `</div>`;
            }
            inMeta = false;
            metaLines = [];
            return;
        }

        // Code blocks
        if (trimmed.startsWith('```')) {
            if (!inCodeBlock) {
                // Start code block
                codeBlockLang = trimmed.slice(3).trim() || 'text';
                codeBlockContent = '';
                inCodeBlock = true;
            } else {
                // End code block
                html += `<div class="article-code-block">
                    <div class="article-code-header">
                        <span>${codeBlockLang}</span>
                        <button class="copy-code-btn text-xs bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded" onclick="copyToClipboard(this)">
                            Copy
                        </button>
                    </div>
                    <div class="article-code-content">
                        <pre><code>${escapeHtml(codeBlockContent)}</code></pre>
                    </div>
                </div>`;
                inCodeBlock = false;
                codeBlockContent = '';
                codeBlockLang = '';
            }
            return;
        }

        if (inCodeBlock) {
            codeBlockContent += line + '\n';
            return;
        }

        // Blockquotes
        if (trimmed.startsWith('> ')) {
            if (!inBlockquote) {
                html += `<blockquote class="article-blockquote">`;
                inBlockquote = true;
            }
            html += `<p>${processInlineFormatting(escapeHtml(trimmed.slice(2)))}</p>`;
            return;
        } else if (inBlockquote && trimmed === '') {
            html += `</blockquote>`;
            inBlockquote = false;
            return;
        }

        // "Poin Penting :" or similar
        if (/^Poin Penting\s*:/.test(trimmed)) {
            html += `<div class="article-section-title">
                <i class="fas fa-star"></i>
                Poin Penting
            </div>`;
            return;
        }

        // "KABARBURSA.COM" or similar source
        if (/^[A-Z0-9\.\-]+$/.test(trimmed) && trimmed.length > 5 && trimmed.length < 30) {
            html += `<div class="article-source">${escapeHtml(trimmed)}</div>`;
            return;
        }

        // "Berita Terkait"
        if (/^Berita Terkait/i.test(trimmed)) {
            beritaTerkait = true;
            html += `<div class="article-section-title">
                <i class="fas fa-newspaper"></i>
                Berita Terkait
            </div>`;
            return;
        }

        // Tables (markdown table support)
        if (trimmed.includes('|') && trimmed.split('|').length > 2) {
            const cells = trimmed.split('|').map(cell => cell.trim()).filter(cell => cell);
            if (cells.length > 1) {
                // Check if this is a separator row (contains only dashes and colons)
                const isSeparator = /^[\s\-\|:]+$/.test(trimmed);
                
                if (!tableOpen && !isSeparator) {
                    // Start new table
                    html += `<div class="article-table-wrapper">
                        <table class="article-table">
                            <thead><tr>`;
                    cells.forEach(cell => {
                        html += `<th>${processInlineFormatting(escapeHtml(cell))}</th>`;
                    });
                    html += `</tr></thead><tbody>`;
                    tableOpen = true;
                } else if (tableOpen && !isSeparator) {
                    // Add data row
                    html += '<tr>';
                    cells.forEach(cell => {
                        html += `<td>${processInlineFormatting(escapeHtml(cell))}</td>`;
                    });
                    html += '</tr>';
                }
            }
            return;
        } else if (tableOpen && trimmed === '') {
            // Close table if a blank line is found after table rows
            html += `</tbody></table></div>`;
            tableOpen = false;
            return;
        }

        // Unordered Lists (bullet points)
        if (/^(–|-|\u2022|\*)\s+(.+)$/.test(trimmed)) {
            if (!inList) {
                html += `<ul class="article-list">`;
                inList = true;
                inOrderedList = false;
            }
            // Remove leading dash/bullet
            let item = trimmed.replace(/^(–|-|\u2022|\*)\s+/, '');
            html += `<li class="article-list-item">${processInlineFormatting(escapeHtml(item))}</li>`;
            return;
        }

        // Ordered Lists (numbered)
        const orderedListMatch = trimmed.match(/^(\d+)\.\s+(.+)$/);
        if (orderedListMatch) {
            if (!inOrderedList) {
                html += `<ol class="article-ordered-list">`;
                inOrderedList = true;
                inList = false;
            }
            const item = orderedListMatch[2];
            html += `<li class="article-ordered-list-item">${processInlineFormatting(escapeHtml(item))}</li>`;
            return;
        }

        // Close lists when empty line is encountered
        if ((inList || inOrderedList) && trimmed === '') {
            if (inList) {
                html += `</ul>`;
                inList = false;
            }
            if (inOrderedList) {
                html += `</ol>`;
                inOrderedList = false;
            }
            beritaTerkait = false;
            return;
        }

        // Images (markdown format)
        const imageMatch = trimmed.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
        if (imageMatch) {
            const alt = imageMatch[1];
            const src = imageMatch[2];
            html += `<div class="article-image-container">
                <img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}" class="article-image" loading="lazy">
                ${alt ? `<div class="article-image-caption">${escapeHtml(alt)}</div>` : ''}
            </div>`;
            return;
        }

        // Image/photo credit
        if (/^Foto\s*:/.test(trimmed)) {
            html += `<div class="article-image-credit">${escapeHtml(trimmed)}</div>`;
            return;
        }

        // Horizontal rules
        if (/^[\s\*\-_]{3,}\s*$/.test(trimmed)) {
            html += `<hr class="article-divider">`;
            return;
        }

        // Section/paragraph
        if (trimmed.length > 0) {
            // Check if this is a special section that should be highlighted
            if (/^(Poin|Point|Key|Important|Penting|Kunci)/i.test(trimmed)) {
                html += `<p class="article-paragraph article-highlight-paragraph">${processInlineFormatting(escapedLine)}</p>`;
            } else {
                html += `<p class="article-paragraph">${processInlineFormatting(escapedLine)}</p>`;
            }
        }
    });

    // Close any open elements
    if (inList) html += `</ul>`;
    if (inOrderedList) html += `</ol>`;
    if (inBlockquote) html += `</blockquote>`;
    if (tableOpen) {
        html += `</tbody></table></div>`;
    }

    html += '</div>';
    return html;
}

// Helper function to escape HTML special characters
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Function to add copy buttons to code blocks
function addCopyButtons($container) {
    const $codeBlocks = $container.find('.code-block-container');
    $codeBlocks.each(function() {
        const $block = $(this);
        const $pre = $block.find('pre');
        if ($pre.length && !$block.find('.copy-button').length) {
            const $button = $('<button>').addClass('copy-button absolute top-2 right-2 bg-gray-700/60 hover:bg-gray-600 text-white p-2 rounded-md text-xs opacity-0 group-hover:opacity-100 transition-opacity duration-200');
            $button.html('<i class="far fa-copy"></i> Copy');

            $button.on('click', function() {
                const code = $pre.find('code').text();
                navigator.clipboard.writeText(code).then(() => {
                    $button.text('Copied!');
                    setTimeout(() => {
                        $button.html('<i class="far fa-copy"></i> Copy');
                    }, 2000);
                }).catch(err => {
                    console.error('Failed to copy text:', err);
                });
            });

            $block.append($button);
            $block.addClass('group');
        }
    });
}

document.addEventListener('DOMContentLoaded', function() {
    // Get the current URL to extract the link parameter
    const currentUrl = window.location.pathname;
    const link = currentUrl.replace('/article-summary/', '');
    
    // Render initial markdown content if it exists
    const initialMarkdownContent = document.getElementById('initial-markdown-content');
    const initialRenderedContent = document.getElementById('initial-rendered-content');
    
    if (initialMarkdownContent && initialRenderedContent) {
        const markdownText = initialMarkdownContent.textContent;
        const renderedContent = renderMarkdown(markdownText);
        initialRenderedContent.innerHTML = renderedContent;
        
        // Add copy buttons and syntax highlighting
        addCopyButtons($(initialRenderedContent));
        if (typeof Prism !== 'undefined') {
            Prism.highlightAllUnder(initialRenderedContent);
        }
    }
    
    // Button click handlers
    document.getElementById('summarizeArticleBtn').addEventListener('click', function() {
        handleButtonClick('Rangkum artikel ini secara mendetail');
    });
    
    document.getElementById('coreArticleBtn').addEventListener('click', function() {
        handleButtonClick('Apa inti atau poin utama dari artikel ini?');
    });
    
    document.getElementById('detailedExplanationBtn').addEventListener('click', function() {
        handleButtonClick('Berikan penjelasan rinci tentang artikel ini');
    });

    document.getElementById('preview-btn').addEventListener('click', function() {
        // Show loading state on the button
        const button = this;
        const originalText = button.innerHTML;
        button.innerHTML = '<span class="animate-spin">⏳</span> Proses...';
        button.disabled = true;

        // Show loading message in content area
        updateContent('Memproses permintaan Anda... Mohon tunggu.');

        // Make AJAX request to the backend for preview (only process_article_with_fallback, no agent)
        fetch(`${window.location.pathname}?query=proses`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Received data from server:', data);

                // Show the PREVIEW result in the preview-artikel-content area
                // and switch to the preview tab
                showPreviewArtikelContent(data);

                // Optionally update source cards with URLs if available
                if (data.urls && data.urls.length > 0) {
                    updateSourceCards(data.urls);
                }

                // Re-enable button
                button.innerHTML = originalText;
                button.disabled = false;
            })
            .catch(error => {
                console.error('Error:', error);
                // Show error message
                updateContent('Terjadi kesalahan saat memproses permintaan. Silakan coba lagi. Jika masalah berlanjut, artikel mungkin tidak dapat diakses.');

                // Re-enable button
                button.innerHTML = originalText;
                button.disabled = false;
            });
    });

    // Helper to show preview result in the preview-artikel-content area and switch tab
    function showPreviewArtikelContent(data) {
        // Switch to the preview tab visually
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.setAttribute('aria-selected', 'false');
            btn.classList.remove('text-indigo-400', 'border-b-2', 'border-indigo-400');
            btn.classList.add('text-gray-400');
        });
        const previewBtn = document.getElementById('preview-btn');
        if (previewBtn) {
            previewBtn.setAttribute('aria-selected', 'true');
            previewBtn.classList.add('text-indigo-400', 'border-b-2', 'border-indigo-400');
            previewBtn.classList.remove('text-gray-400');
        }

        // Hide all tab panels
        document.querySelectorAll('[role="tabpanel"]').forEach(panel => {
            panel.classList.add('hidden');
        });

        // Show the preview-artikel-content panel
        const previewPanel = document.getElementById('preview-artikel-content');
        if (previewPanel) {
            previewPanel.classList.remove('hidden');
            const previewResultContent = document.getElementById('preview-result-content');
            if (previewResultContent) {
                // Enhanced Article Renderer with CSS Modules and Responsive Design
                function renderStyledArticle(text) {
                    // Split by double newlines for paragraphs/sections
                    const lines = text.split('\n');
                    let html = '<div class="article-container article-fade-in">';
                    let inList = false;
                    let inMeta = false;
                    let metaLines = [];
                    let beritaTerkait = false;
                    let inBlockquote = false;
                    let inCodeBlock = false;
                    let codeBlockContent = '';
                    let codeBlockLang = '';

                    // Helper function to escape HTML
                    function escapeHtml(text) {
                        const div = document.createElement('div');
                        div.textContent = text;
                        return div.innerHTML;
                    }

                                         // Helper function to process inline formatting
                     function processInlineFormatting(text) {
                         return text
                             // Bold text (handle nested formatting)
                             .replace(/\*\*(.*?)\*\*/g, '<strong class="article-strong">$1</strong>')
                             // Italic text (handle nested formatting)
                             .replace(/\*(.*?)\*/g, '<em class="article-emphasis">$1</em>')
                             // Inline code
                             .replace(/`(.*?)`/g, '<code class="article-inline-code">$1</code>')
                             // Links
                             .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="article-link">$1</a>')
                             // Highlight important terms
                             .replace(/\b(IMPORTANT|PENTING|KRITIS|URGENT|PENTING|KRITIS|URGENT)\b/gi, '<span class="article-highlight">$1</span>')
                             // Highlight numbers and percentages
                             .replace(/(\d+(?:\.\d+)?%?)/g, '<span class="article-number">$1</span>')
                             // Highlight currency
                             .replace(/(Rp\.?\s*\d+(?:,\d{3})*(?:\.\d{2})?)/gi, '<span class="article-currency">$1</span>')
                             // Highlight email addresses
                             .replace(/([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})/g, '<a href="mailto:$1" class="article-link">$1</a>')
                             // Highlight URLs (basic)
                             .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer" class="article-link">$1</a>');
                     }

                    lines.forEach((line, idx) => {
                        const trimmed = line.trim();
                        const escapedLine = escapeHtml(trimmed);

                        // Headline (first non-empty line)
                        if (idx === 0 && trimmed.length > 0) {
                            html += `<h1 class="article-headline">${processInlineFormatting(escapedLine)}</h1>`;
                            return;
                        }

                        // Byline (Ditulis oleh ...)
                        if (/^Ditulis oleh/i.test(trimmed)) {
                            html += `<div class="article-byline">
                                <i class="fas fa-user-edit article-byline-icon"></i>
                                <span>${processInlineFormatting(escapedLine)}</span>
                            </div>`;
                            inMeta = true;
                            metaLines = [];
                            return;
                        }

                        // Date or A+/A- (meta info)
                        if (inMeta && (/^\d{1,2}\s+\w+\s+\d{4}$/.test(trimmed) || trimmed === 'A+' || trimmed === 'A−' || trimmed === 'A-')) {
                            metaLines.push(trimmed);
                            return;
                        }
                        if (inMeta && trimmed === '') {
                            // End of meta block
                            if (metaLines.length > 0) {
                                html += `<div class="article-meta">`;
                                metaLines.forEach((m, i) => {
                                    if (i > 0) html += '<span class="mx-1">•</span>';
                                    html += `<span class="article-meta-item">${escapeHtml(m)}</span>`;
                                });
                                html += `</div>`;
                            }
                            inMeta = false;
                            metaLines = [];
                            return;
                        }

                        // Code blocks
                        if (trimmed.startsWith('```')) {
                            if (!inCodeBlock) {
                                // Start code block
                                codeBlockLang = trimmed.slice(3).trim() || 'text';
                                codeBlockContent = '';
                                inCodeBlock = true;
                            } else {
                                // End code block
                                html += `<div class="article-code-block">
                                    <div class="article-code-header">
                                        <span>${codeBlockLang}</span>
                                        <button class="copy-code-btn text-xs bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded" onclick="copyToClipboard(this)">
                                            Copy
                                        </button>
                                    </div>
                                    <div class="article-code-content">
                                        <pre><code>${escapeHtml(codeBlockContent)}</code></pre>
                                    </div>
                                </div>`;
                                inCodeBlock = false;
                                codeBlockContent = '';
                                codeBlockLang = '';
                            }
                            return;
                        }

                        if (inCodeBlock) {
                            codeBlockContent += line + '\n';
                            return;
                        }

                        // Blockquotes
                        if (trimmed.startsWith('> ')) {
                            if (!inBlockquote) {
                                html += `<blockquote class="article-blockquote">`;
                                inBlockquote = true;
                            }
                            html += `<p>${processInlineFormatting(escapeHtml(trimmed.slice(2)))}</p>`;
                            return;
                        } else if (inBlockquote && trimmed === '') {
                            html += `</blockquote>`;
                            inBlockquote = false;
                            return;
                        }

                        // "Poin Penting :" or similar
                        if (/^Poin Penting\s*:/.test(trimmed)) {
                            html += `<div class="article-section-title">
                                <i class="fas fa-star"></i>
                                Poin Penting
                            </div>`;
                            return;
                        }

                        // "KABARBURSA.COM" or similar source
                        if (/^[A-Z0-9\.\-]+$/.test(trimmed) && trimmed.length > 5 && trimmed.length < 30) {
                            html += `<div class="article-source">${escapeHtml(trimmed)}</div>`;
                            return;
                        }

                        // "Berita Terkait"
                        if (/^Berita Terkait/i.test(trimmed)) {
                            beritaTerkait = true;
                            html += `<div class="article-section-title">
                                <i class="fas fa-newspaper"></i>
                                Berita Terkait
                            </div>`;
                            return;
                        }

                        // Tables (basic markdown table support)
                        if (trimmed.includes('|') && trimmed.split('|').length > 2) {
                            const cells = trimmed.split('|').map(cell => cell.trim()).filter(cell => cell);
                            if (cells.length > 1) {
                                if (!html.includes('<table class="article-table">')) {
                                    html += `<div class="article-table-wrapper">
                                        <table class="article-table">
                                            <thead><tr>`;
                                    cells.forEach(cell => {
                                        html += `<th>${processInlineFormatting(escapeHtml(cell))}</th>`;
                                    });
                                    html += `</tr></thead><tbody>`;
                                } else {
                                    html += '<tr>';
                                    cells.forEach(cell => {
                                        html += `<td>${processInlineFormatting(escapeHtml(cell))}</td>`;
                                    });
                                    html += '</tr>';
                                }
                            }
                            return;
                        }

                        // List points (after "Poin Penting" or "Berita Terkait" or with bullets)
                        if ((beritaTerkait || /^(–|-|\u2022|\*)/.test(trimmed)) && trimmed.length > 2) {
                            if (!inList) {
                                html += `<ul class="article-list">`;
                                inList = true;
                            }
                            // Remove leading dash/bullet
                            let item = trimmed.replace(/^(–|-|\u2022|\*)\s*/, '');
                            html += `<li class="article-list-item">${processInlineFormatting(escapeHtml(item))}</li>`;
                            return;
                        } else if (inList && trimmed === '') {
                            html += `</ul>`;
                            inList = false;
                            beritaTerkait = false;
                            return;
                        }

                        // Image/photo credit
                        if (/^Foto\s*:/.test(trimmed)) {
                            html += `<div class="article-image-credit">${escapeHtml(trimmed)}</div>`;
                            return;
                        }

                        // Horizontal rules
                        if (/^[\s\*\-_]{3,}\s*$/.test(trimmed)) {
                            html += `<hr class="article-divider">`;
                            return;
                        }

                        // Section/paragraph
                        if (trimmed.length > 0) {
                            html += `<p class="article-paragraph">${processInlineFormatting(escapedLine)}</p>`;
                        }
                    });

                    // Close any open elements
                    if (inList) html += `</ul>`;
                    if (inBlockquote) html += `</blockquote>`;
                    if (html.includes('<table class="article-table">')) {
                        html += `</tbody></table></div>`;
                    }

                    html += '</div>';
                    return html;
                }

                // Copy to clipboard function
                function copyToClipboard(button) {
                    const codeBlock = button.closest('.article-code-block');
                    const code = codeBlock.querySelector('code').textContent;
                    
                    navigator.clipboard.writeText(code).then(() => {
                        button.textContent = 'Copied!';
                        setTimeout(() => {
                            button.textContent = 'Copy';
                        }, 2000);
                    }).catch(err => {
                        console.error('Failed to copy text:', err);
                        button.textContent = 'Error';
                        setTimeout(() => {
                            button.textContent = 'Copy';
                        }, 2000);
                    });
                }

                if (data.result && data.result.status === 'success' && data.result.full_text) {
                    previewResultContent.innerHTML = `
                        <div class="article-preview-container">
                          <h2 class="text-lg font-bold mb-4 text-indigo-300 flex items-center gap-2">
                            <i class="fas fa-eye"></i> Preview Artikel
                          </h2>
                          <div class="bg-gray-900/80 rounded-xl p-6 text-gray-200 overflow-x-auto whitespace-normal max-w-none border border-gray-700/50 shadow-xl">
                            ${renderStyledArticle(data.result.full_text)}
                          </div>
                        </div>
                    `;
                } else if (data.result && data.result.summary) {
                    previewResultContent.innerHTML = `
                        <div class="article-preview-container">
                          <h2 class="text-lg font-bold mb-4 text-indigo-300 flex items-center gap-2">
                            <i class="fas fa-list-alt"></i> Ringkasan Artikel
                          </h2>
                          <div class="bg-gray-900/80 rounded-xl p-6 text-gray-200 overflow-x-auto whitespace-normal max-w-none border border-gray-700/50 shadow-xl">
                            ${renderStyledArticle(data.result.summary)}
                          </div>
                        </div>
                    `;
                } else {
                    previewResultContent.innerHTML = `
                        <div class="text-center text-gray-400 py-12">
                          <i class="fas fa-file-alt text-4xl mb-4 opacity-50"></i>
                          <p class="text-lg">Tidak ada data hasil proses artikel untuk ditampilkan.</p>
                        </div>
                    `;
                }
            }
        }
        
    }
    
    function handleButtonClick(query) {
        // Show loading state
        const button = event.target.closest('button');
        const originalText = button.innerHTML;
        button.innerHTML = '<span class="animate-spin">⏳</span> Processing...';
        button.disabled = true;
        
        // Show loading message in content area
        updateContent('Processing your request... Please wait.');
        
        // Make AJAX request to the backend
        fetch(`${currentUrl}?query=${encodeURIComponent(query)}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Received data from server:', data);
                
                // Update the content area with the response
                if (data.agent_response) {
                    updateContent(data.agent_response);
                } else {
                    updateContent('No response received from the server.');
                }
                
                // Update source cards with URLs if available
                if (data.urls && data.urls.length > 0) {
                    console.log('Updating source cards with URLs:', data.urls);
                    updateSourceCards(data.urls);
                } else {
                    console.log('No URLs found in response');
                }
                
                // Re-enable button
                button.innerHTML = originalText;
                button.disabled = false;
            })
            .catch(error => {
                console.error('Error:', error);
                // Show error message
                updateContent('Error processing request. Please try again. If the problem persists, the article might not be accessible.');
                
                // Re-enable button
                button.innerHTML = originalText;
                button.disabled = false;
            });
    }
    
    function updateContent(content) {
        // Update the answer content area
        const answerContent = document.getElementById('answer-content');
        const contentArea = answerContent.querySelector('.space-y-4');
        
        // Render markdown content
        const renderedContent = renderMarkdown(content);
        
        // Replace the existing content with the new response
        contentArea.innerHTML = `
            <div class="space-y-4 text-gray-300 leading-relaxed article-fade-in">
                ${renderedContent}
            </div>
        `;
        
        // Add copy buttons to code blocks
        addCopyButtons($(contentArea));
        
        // Highlight code blocks with Prism.js
        if (typeof Prism !== 'undefined') {
            Prism.highlightAllUnder(contentArea);
        }
        
        // Add smooth scrolling to the content
        contentArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    function updateSourceCards(urls) {
        console.log('updateSourceCards called with:', urls);
        
        // Update the source cards container with the new URLs
        const container = document.getElementById('source-cards-container');
        const sourcesList = document.getElementById('sources-list');
        
        console.log('Container elements found:', { container: !!container, sourcesList: !!sourcesList });
        
        // Clear existing cards
        container.innerHTML = '';
        sourcesList.innerHTML = '';
        
        // Add new cards for each URL
        urls.forEach((urlData, index) => {
            // Create card for Answer tab
            const card = document.createElement('a');
            card.href = urlData.url;
            card.target = '_blank';
            card.className = 'block bg-gray-900/80 p-3 sm:p-4 rounded-xl shadow hover:scale-[1.03] hover:bg-indigo-900/60 transition-all duration-200 group focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 w-full';
            
            card.innerHTML = `
                <div class="flex flex-col sm:flex-row items-start sm:items-center mb-2 gap-2">
                    <img src="https://placehold.co/24x24/4285F4/FFFFFF?text=${urlData.index}" alt="Favicon" class="w-6 h-6 sm:w-4 sm:h-4 mr-0 sm:mr-2 rounded">
                    <span class="text-xs sm:text-sm text-gray-400 break-words">${urlData.title}</span>
                </div>
            `;
            
            container.appendChild(card);
            
            // Create list item for Sources tab
            const listItem = document.createElement('li');
            listItem.className = 'flex items-start gap-4';
            
            listItem.innerHTML = `
                <span class="inline-flex items-center justify-center w-5 h-5 bg-gray-800 rounded text-xs text-gray-400 mt-1">${urlData.index}</span>
                <a href="${urlData.url}" target="_blank" class="group focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400">
                    <p class="text-white group-hover:underline">${urlData.title}</p>
                    <div class="flex items-center text-xs text-gray-400 mt-1">
                        <img src="https://placehold.co/16x16/4285F4/FFFFFF?text=${urlData.index}" alt="Favicon" class="w-4 h-4 mr-2 rounded">
                        <span>${urlData.url}</span>
                    </div>
                </a>
            `;
            
            sourcesList.appendChild(listItem);
        });
        
        // Update the sources tab count
        const sourcesBtn = document.getElementById('sources-btn');
        const countSpan = sourcesBtn.querySelector('span:last-child');
        if (countSpan) {
            countSpan.textContent = urls.length;
        }
    }
    
    // Tab functionality
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('[role="tabpanel"]');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetPanel = this.getAttribute('aria-controls');
            
            // Update button states
            tabButtons.forEach(btn => {
                btn.classList.remove('text-indigo-400', 'border-b-2', 'border-indigo-400');
                btn.classList.add('text-gray-400', 'hover:text-indigo-300');
                btn.setAttribute('aria-selected', 'false');
            });
            
            this.classList.remove('text-gray-400', 'hover:text-indigo-300');
            this.classList.add('text-indigo-400', 'border-b-2', 'border-indigo-400');
            this.setAttribute('aria-selected', 'true');
            
            // Update panel visibility
            tabPanels.forEach(panel => {
                panel.classList.add('hidden');
            });
            
            document.getElementById(targetPanel).classList.remove('hidden');
        });
    });
});