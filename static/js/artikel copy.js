// Markdown rendering function
function renderMarkdown(markdownText) {
    // Escape HTML entities first to prevent unwanted HTML rendering
    let html = escapeHtml(markdownText);

    // Store code blocks and replace with placeholders
    const codeBlocks = [];
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        const placeholder = `CODE_BLOCK_PLACEHOLDER_${codeBlocks.length}__`;
        codeBlocks.push({ lang: lang, code: code });
        return placeholder;
    });

    // Wrap the entire HTML in a div with the prose class
    html = `<div class="prose dark:prose-invert max-w-none text-white">${html}</div>`;

    // Horizontal Rule
    html = html.replace(/^[\s\*\-_]{3,}\s*$/gm, '<hr>');

    // Headers (H1-H6) - no inline classes needed, handled by prose
    html = html.replace(/^###### (.*)$/gm, '<h6>$1</h6>');
    html = html.replace(/^##### (.*)$/gm, '<h5>$1</h5>');
    html = html.replace(/^#### (.*)$/gm, '<h4>$1</h4>');
    html = html.replace(/^### (.*)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.*)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.*)$/gm, '<h1>$1</h1>');

    // Blockquotes - no inline classes needed, handled by prose
    html = html.replace(/^> (.*)$/gm, '<blockquote>$1</blockquote>');

    // Lists (unordered and ordered) - more robust handling
    html = html.replace(/\n(?= *- |^\d+\. )/g, '@@NEWLINE_HOLDER@@');
    // Unordered lists
    // Convert markdown list items to <li> tags
    html = html.replace(/^- (.*)$/gm, '<li>$1</li>');

    // Ordered lists
    // Convert markdown ordered list items to <li> tags, keeping the number in the text
    html = html.replace(/^(\d+)\. (.*)$/gm, '<li>$1. $2</li>');

    // Wrap consecutive <li> elements into a single <ul> or <ol> with list-style: none
    // This regex looks for one or more lines that start with <li> and end with </li>.
    // The \n? allows for a single newline between list items, ensuring blocks are correctly grouped.
    // It differentiates between ordered and unordered lists based on the content of the first li.
    html = html.replace(/((?:<li>.*?<\/li>\n?)+)/g, (match, listItems) => {
        // Remove any trailing newlines from the captured listItems block before wrapping
        listItems = listItems.trim();
        // Check if the list is ordered by looking for a number followed by a period in the first item
        const isOrdered = /^\\d+\\./.test(listItems);
        if (isOrdered) {
            return `<ol class="list-none">${listItems}</ol>`;
        } else {
            return `<ul class="list-none">${listItems}</ul>`;
        }
    });
    html = html.replace(/@@NEWLINE_HOLDER@@/g, '<br>');

    // Links - no inline classes needed, handled by prose
    html = html.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

    // Images - no inline classes needed, handled by prose
    html = html.replace(/!\[(.*?)\]\((.*?)\)/g, '<img src="$2" alt="$1">');

    // Tables (basic: assumes header and at least one row, no alignment parsing) - no inline classes needed, handled by prose
    html = html.replace(/\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\n\|\s*---+\s*\|\s*---+\s*\|\s*---+\s*\|\n((?:\|\s*.*?\s*\|\s*.*?\s*\|\s*.*?\s*\|\n)+)/g, (match, header1, header2, header3, rows) => {
        let tableHtml = '<table><thead><tr>';
        tableHtml += `<th>${header1}</th>`;
        tableHtml += `<th>${header2}</th>`;
        tableHtml += `<th>${header3}</th>`;
        tableHtml += '</tr></thead><tbody>';

        rows.trim().split('\n').forEach(row => {
            const cols = row.split('|').map(c => c.trim()).filter(c => c);
            if (cols.length === 3) {
                tableHtml += '<tr>';
                tableHtml += `<td>${cols[0]}</td>`;
                tableHtml += `<td>${cols[1]}</td>`;
                tableHtml += `<td>${cols[2]}</td>`;
                tableHtml += '</tr>';
            }
        });

        tableHtml += '</tbody></table>';
        return tableHtml;
    });

    // Math Notation (basic: inline $...$ and block $$...$$) - no inline classes needed, handled by prose
    html = html.replace(/\$\$(.*?)\$\$/gs, '<div class="math-block">$1</div>');
    html = html.replace(/\$(.*?)\$/g, '<span class="math-inline">$1</span>');

    // Inline code - no inline classes needed, handled by prose
    html = html.replace(/`(.*?)`/g, '<code>$1</code>');

    // Newlines to <br> (this should be after block-level elements are handled)
    html = html.replace(/\n/g, '<br>');

    // Basic bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>');

    // Basic italics
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Restore code blocks
    for (let i = 0; i < codeBlocks.length; i++) {
        const placeholder = `CODE_BLOCK_PLACEHOLDER_${i}__`;
        const block = codeBlocks[i];
        const languageClass = block.lang ? `language-${block.lang}` : 'language-markup';
        const codeHtml = `<pre><code class="${languageClass}">${block.code}</code></pre>`;
        html = html.replace(placeholder, codeHtml);
    }

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
                // Helper: parse the plain text response into styled blocks
                function renderStyledArticle(text) {
                    // Split by double newlines for paragraphs/sections
                    const lines = text.split('\n');
                    let html = '';
                    let inList = false;
                    let inMeta = false;
                    let metaLines = [];
                    let beritaTerkait = false;

                    lines.forEach((line, idx) => {
                        const trimmed = line.trim();

                        // Headline (first non-empty line)
                        if (idx === 0 && trimmed.length > 0) {
                            html += `<h1 class="text-2xl font-bold mb-2 text-white">${trimmed}</h1>`;
                            return;
                        }

                        // Byline (Ditulis oleh ...)
                        if (/^Ditulis oleh/i.test(trimmed)) {
                            html += `<div class="flex items-center gap-2 text-sm text-gray-400 mb-1 mt-2"><i class="fas fa-user-edit"></i> <span>${trimmed}</span></div>`;
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
                                html += `<div class="flex items-center gap-2 text-xs text-gray-500 mb-3">`;
                                metaLines.forEach((m, i) => {
                                    // if (i > 0) html += '<span class="mx-1">•</span>';
                                    // html += `<span>${m}</span>`;
                                });
                                html += `</div>`;
                            }
                            inMeta = false;
                            metaLines = [];
                            return;
                        }

                        // "Poin Penting :" or similar
                        if (/^Poin Penting\s*:/.test(trimmed)) {
                            html += `<div class="mt-4 mb-2 text-base font-semibold text-indigo-400">Poin Penting</div>`;
                            return;
                        }

                        // "KABARBURSA.COM" or similar source
                        if (/^[A-Z0-9\.\-]+$/.test(trimmed) && trimmed.length > 5 && trimmed.length < 30) {
                            html += `<div class="text-xs font-bold text-blue-400 mb-2">${trimmed}</div>`;
                            return;
                        }

                        // "Berita Terkait"
                        if (/^Berita Terkait/i.test(trimmed)) {
                            beritaTerkait = true;
                            html += `<div class="mt-6 mb-2 text-base font-semibold text-indigo-400">Berita Terkait</div>`;
                            return;
                        }

                        // List points (after "Poin Penting" or "Berita Terkait")
                        if ((beritaTerkait || /^(–|-|\u2022)/.test(trimmed)) && trimmed.length > 2) {
                            if (!inList) {
                                html += `<ul class="list-disc pl-6 mb-2 text-gray-200">`;
                                inList = true;
                            }
                            // Remove leading dash/bullet
                            let item = trimmed.replace(/^(–|-|\u2022)\s*/, '');
                            html += `<li class="mb-1">${item}</li>`;
                            return;
                        } else if (inList && trimmed === '') {
                            html += `</ul>`;
                            inList = false;
                            beritaTerkait = false;
                            return;
                        }

                        // Image/photo credit
                        if (/^Foto\s*:/.test(trimmed)) {
                            html += `<div class="text-xs italic text-gray-500 mb-2">${trimmed}</div>`;
                            return;
                        }

                        // Section/paragraph
                        if (trimmed.length > 0) {
                            html += `<p class="mb-3 text-gray-200 leading-relaxed">${trimmed}</p>`;
                        }
                    });
                    if (inList) html += `</ul>`;
                    return html;
                }

                if (data.result && data.result.status === 'success' && data.result.full_text) {
                    previewResultContent.innerHTML = `
                        <div>
                          <h2 class="text-lg font-bold mb-3 text-indigo-300 flex items-center gap-2">
                            <i class="fas fa-eye"></i> Preview Artikel
                          </h2>
                          <div class="bg-gray-900/80 rounded-xl p-4 text-gray-200 overflow-x-auto whitespace-normal prose prose-invert max-w-none">
                            ${renderStyledArticle(data.result.full_text)}
                          </div>
                        </div>
                    `;
                } else if (data.result && data.result.summary) {
                    previewResultContent.innerHTML = `
                        <div>
                          <h2 class="text-lg font-bold mb-3 text-indigo-300 flex items-center gap-2">
                            <i class="fas fa-list-alt"></i> Ringkasan Artikel
                          </h2>
                          <div class="bg-gray-900/80 rounded-xl p-4 text-gray-200 overflow-x-auto whitespace-normal prose prose-invert max-w-none">
                            ${renderStyledArticle(data.result.summary)}
                          </div>
                        </div>
                    `;
                } else {
                    previewResultContent.innerHTML = `
                        <p class="text-center text-gray-400 py-8">
                          Tidak ada data hasil proses artikel untuk ditampilkan.
                        </p>
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
            <div class="space-y-4 text-white leading-relaxed">
                ${renderedContent}
            </div>
        `;
        
        // Add copy buttons to code blocks
        addCopyButtons($(contentArea));
        
        // Highlight code blocks with Prism.js
        if (typeof Prism !== 'undefined') {
            Prism.highlightAllUnder(contentArea);
        }
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