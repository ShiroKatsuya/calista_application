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
    
    // Advanced Markdown to HTML converter with Tailwind classes (same as multi_agent.js)
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
        html = `<div class="prose dark:prose-invert max-w-none">${html}</div>`;

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
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

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