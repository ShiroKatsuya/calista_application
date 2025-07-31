$(document).ready(function() {
    // Process existing conversation messages on page load
    processExistingMessages();
    
    // Handle clear input button for follow-up input
    const $followUpInput = $('#followUpInput');
    const $clearInputButton = $('#clearInputButton');
    
    function toggleClearButton() {
        if ($followUpInput.val().trim() !== '') {
            $clearInputButton.show();
        } else {
            $clearInputButton.hide();
        }
    }
    
    $followUpInput.on('input', toggleClearButton);
    $clearInputButton.on('click', function() {
        $followUpInput.val('');
        toggleClearButton();
        $followUpInput.focus();
    });
    
    // Function to process existing messages and apply styling
    function processExistingMessages() {
        const $responseContainer = $('#responseContainer');
        if (!$responseContainer.length) return;
        
        // Process each message in the container
        $responseContainer.find('.message-bubble').each(function() {
            const $messageDiv = $(this);
            const $messageContent = $messageDiv.find('.message-content');
            
            if ($messageContent.length) {
                const rawContent = $messageContent.html();
                const processedContent = renderMarkdown(rawContent);
                $messageContent.html(processedContent);
                
                // Add copy buttons to code blocks
                addCopyButtons($messageDiv);
                
                // Add action buttons
                const $actionButtonsContainer = createActionButtons();
                $messageDiv.append($actionButtonsContainer);
                
                // Highlight code blocks
                Prism.highlightAllUnder($messageDiv[0]);
            }
        });
    }
    

// Advanced Markdown to HTML converter with CSS Modules
function renderMarkdown(markdownText) {
    // 1. First, handle <think> blocks.
    // We'll replace them with placeholders, and for their content,
    // we'll use the _renderThinkContentMarkdown helper.
    const thinkBlockData = [];
    const tempHtml = markdownText.replace(/<think>([\s\S]*?)<\/think>/g, (match, content) => {
        const placeholder = `__THINK_BLOCK_${thinkBlockData.length}__`;
        thinkBlockData.push(content);
        return placeholder;
    });

    // 2. Now, escape HTML for the rest of the text that's not part of <think>
    let html = escapeHtml(tempHtml);

    // Store code blocks and replace with placeholders
    const codeBlocks = [];
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        const placeholder = `CODE_BLOCK_PLACEHOLDER_${codeBlocks.length}__`;
        codeBlocks.push({ lang: lang, code: code });
        return placeholder;
    });

    // Horizontal Rule
    html = html.replace(/^[\s\*\-_]{3,}\s*$/gm, '<hr class="markdown-hr">');

    // Headers (H1-H6) with CSS Module classes
    html = html.replace(/^###### (.*)$/gm, '<h6 class="markdown-h6">$1</h6>');
    html = html.replace(/^##### (.*)$/gm, '<h5 class="markdown-h5">$1</h5>');
    html = html.replace(/^#### (.*)$/gm, '<h4 class="markdown-h4">$1</h4>');
    html = html.replace(/^### (.*)$/gm, '<h3 class="markdown-h3">$1</h3>');
    html = html.replace(/^## (.*)$/gm, '<h2 class="markdown-h2">$1</h2>');
    html = html.replace(/^# (.*)$/gm, '<h1 class="markdown-h1">$1</h1>');

    // Blockquotes
    html = html.replace(/^> (.*)$/gm, '<blockquote class="markdown-blockquote"><p class="markdown-p">$1</p></blockquote>');

    // Lists (unordered and ordered) - more robust handling
    html = html.replace(/\n(?= *- |^\d+\. )/g, '@@NEWLINE_HOLDER@@');
    // Unordered lists
    html = html.replace(/^- (.*)$/gm, '<li class="markdown-li">$1</li>');
    // Ordered lists
    html = html.replace(/^(\d+)\. (.*)$/gm, '<li class="markdown-li">$1. $2</li>');
    // Wrap consecutive <li> elements into a single <ul> with CSS Module classes
    html = html.replace(/((?:<li class="markdown-li">.*?<\/li>\n?)+)/g, (match, listItems) => {
        listItems = listItems.trim();
        return `<ul class="markdown-ul">${listItems}</ul>`;
    });
    html = html.replace(/@@NEWLINE_HOLDER@@/g, '<br>');

    // Links
    html = html.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="markdown-a">$1</a>');

    // Images
    html = html.replace(/!\[(.*?)\]\((.*?)\)/g, '<img src="$2" alt="$1" class="markdown-img">');

    // Enhanced Table parsing with support for various table formats
    html = html.replace(/((?:\|.*\|(?:\r?\n|\r)?)+)/g, (match) => {
        const lines = match.trim().split(/\r?\n/);
        if (lines.length < 2) return match; // Need at least header and separator
        
        // Check if it's a proper markdown table (has separator line with dashes)
        const hasSeparator = lines.some(line => 
            line.trim().match(/^\|[\s\-:|]+\|$/)
        );
        
        if (!hasSeparator) return match; // Not a table
        
        let tableHtml = '<div class="markdown-table-wrapper"><table class="markdown-table">';
        let inHeader = true;
        let headerAdded = false;
        
        lines.forEach((line, index) => {
            line = line.trim();
            if (!line.startsWith('|') || !line.endsWith('|')) return;
            
            // Skip separator lines (lines with only dashes, colons, and pipes)
            if (line.match(/^\|[\s\-:|]+\|$/)) {
                inHeader = false;
                return;
            }
            
            // Parse cells
            const cells = line.split('|').slice(1, -1).map(cell => cell.trim());
            
            if (inHeader && !headerAdded) {
                tableHtml += '<thead><tr>';
                cells.forEach(cell => {
                    tableHtml += `<th>${escapeHtml(cell)}</th>`;
                });
                tableHtml += '</tr></thead><tbody>';
                headerAdded = true;
            } else if (!inHeader) {
                tableHtml += '<tr>';
                cells.forEach(cell => {
                    tableHtml += `<td>${escapeHtml(cell)}</td>`;
                });
                tableHtml += '</tr>';
            }
        });
        
        tableHtml += '</tbody></table></div>';
        return tableHtml;
    });

    // Math Notation (basic: inline $...$ and block $$...$$)
    html = html.replace(/\$\$(.*?)\$\$/gs, '<div class="markdown-math-block">$1</div>');
    html = html.replace(/\$(.*?)\$/g, '<span class="markdown-math-inline">$1</span>');

    // Inline code
    html = html.replace(/`(.*?)`/g, '<code class="markdown-code-inline">$1</code>');

    // Newlines to <br> (this should be after block-level elements are handled)
    html = html.replace(/\n/g, '<br>');

    // Basic bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="markdown-strong">$1</strong>');

    // Basic italics
    html = html.replace(/\*(.*?)\*/g, '<em class="markdown-em">$1</em>');

    // Restore code blocks
    for (let i = 0; i < codeBlocks.length; i++) {
        const placeholder = `CODE_BLOCK_PLACEHOLDER_${i}__`;
        const block = codeBlocks[i];
        const languageClass = block.lang ? `language-${block.lang}` : 'language-markup';
        // Don't escape code content - we want to display the actual characters
        const codeHtml = `<pre class="markdown-pre"><code class="markdown-code ${languageClass}">${block.code}</code></pre>`;
        html = html.replace(placeholder, codeHtml);
    }

    // 4. Finally, restore the <think> blocks and recursively render their content
    for (let i = 0; i < thinkBlockData.length; i++) {
        const placeholder = `__THINK_BLOCK_${i}__`;
        // Recursively render markdown for the content inside <think>
        const renderedThinkContent = _renderThinkContentMarkdown(thinkBlockData[i]);
        const thinkHtml = `<div class="markdown-thought-process"><details><summary class="markdown-thought-summary">Thought Process</summary><div class="markdown-thought-content">${renderedThinkContent}</div></details></div>`;
        html = html.replace(placeholder, thinkHtml);
    }

    // Wrap the entire content in a markdown container
    return `<div class="markdown-container markdown-fade-in">${html}</div>`;
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

    // Function to create and return a container with action buttons
    function createActionButtons() {
        const $buttonsContainer = $('<div>').addClass('action-buttons flex mt-4 space-x-3 ml-auto pr-2');

        const buttonConfigs = [
            { icon: 'fas fa-thumbs-up', label: 'Good Response', className: 'good-response-button' },
            { icon: 'fas fa-thumbs-down', label: 'Bad Response', className: 'bad-response-button' },
            { icon: 'fas fa-volume-up', label: 'Read Aloud', className: 'read-aloud-button' },
            { icon: 'fas fa-edit', label: 'Edit Canvas', className: 'edit-canvas-button' },
            { icon: 'fas fa-share-alt', label: 'Share', className: 'share-button' }
        ];

        buttonConfigs.forEach(config => {
            const $button = $('<button>').addClass(`action-button ${config.className} text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1`);
            $button.html(`<i class="${config.icon}"></i><span>${config.label}</span>`);
            $buttonsContainer.append($button);
        });

        return $buttonsContainer;
    }
}); 