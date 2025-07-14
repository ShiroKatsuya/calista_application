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

        // Horizontal Rule
        html = html.replace(/^[\s\*\-_]{3,}\s*$/gm, '<hr class="my-6 border-t-2 border-gray-600">');

        // Headers (H1-H6)
        html = html.replace(/^###### (.*)$/gm, '<h6 class="text-sm font-semibold mt-4 mb-2 text-gray-300">$1</h6>');
        html = html.replace(/^##### (.*)$/gm, '<h5 class="text-base font-semibold mt-4 mb-2 text-gray-300">$1</h5>');
        html = html.replace(/^#### (.*)$/gm, '<h4 class="text-lg font-semibold mt-4 mb-2 text-gray-300">$1</h4>');
        html = html.replace(/^### (.*)$/gm, '<h3 class="text-xl font-semibold mt-4 mb-3 text-gray-200">$1</h3>');
        html = html.replace(/^## (.*)$/gm, '<h2 class="text-2xl font-bold mt-5 mb-3 text-gray-100">$1</h2>');
        html = html.replace(/^# (.*)$/gm, '<h1 class="text-3xl font-bold mt-6 mb-4 text-white">$1</h1>');

        // Blockquotes
        html = html.replace(/^> (.*)$/gm, '<blockquote class="border-l-4 border-blue-500 pl-4 py-1 italic text-gray-300 my-4">$1</blockquote>');

        // Lists (unordered and ordered) - more robust handling
        html = html.replace(/\n(?= *- |^\d+\. )/g, '@@NEWLINE_HOLDER@@');
        // Unordered lists
        html = html.replace(/^- (.*)$/gm, '<li class="mb-1">$1</li>');

        // Ordered lists
        html = html.replace(/^(\d+)\. (.*)$/gm, '<li class="mb-1">$1. $2</li>');

        // Wrap consecutive <li> elements into a single <ul>
        html = html.replace(/((?:<li class="mb-1">.*?<\/li>\n?)+)/g, (match, listItems) => {
            listItems = listItems.trim();
            return `<ul class="list-none pl-5 mb-2 text-gray-200">${listItems}</ul>`;
        });
        html = html.replace(/@@NEWLINE_HOLDER@@/g, '<br>');

        // Links
        html = html.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-blue-400 hover:underline">$1</a>');

        // Images
        html = html.replace(/!\[(.*?)\]\((.*?)\)/g, '<img src="$2" alt="$1" class="max-w-full h-auto rounded-lg my-4 shadow-lg">');

        // Tables (basic: assumes header and at least one row, no alignment parsing)
        html = html.replace(/\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\n\|\s*---+\s*\|\s*---+\s*\|\s*---+\s*\|\n((?:\|\s*.*?\s*\|\s*.*?\s*\|\s*.*?\s*\|\n)+)/g, (match, header1, header2, header3, rows) => {
            let tableHtml = '<div class="overflow-x-auto my-4"><table class="min-w-full divide-y divide-gray-600 rounded-lg overflow-hidden">';
            tableHtml += '<thead class="bg-gray-700"><tr class="text-left text-gray-200">';
            tableHtml += `<th class="py-2 px-4 font-semibold">${header1}</th>`;
            tableHtml += `<th class="py-2 px-4 font-semibold">${header2}</th>`;
            tableHtml += `<th class="py-2 px-4 font-semibold">${header3}</th>`;
            tableHtml += '</tr></thead>';
            tableHtml += '<tbody class="bg-gray-800 divide-y divide-gray-700">';

            rows.trim().split('\n').forEach(row => {
                const cols = row.split('|').map(c => c.trim()).filter(c => c);
                if (cols.length === 3) {
                    tableHtml += '<tr class="text-gray-200">';
                    tableHtml += `<td class="py-2 px-4">${cols[0]}</td>`;
                    tableHtml += `<td class="py-2 px-4">${cols[1]}</td>`;
                    tableHtml += `<td class="py-2 px-4">${cols[2]}</td>`;
                    tableHtml += '</tr>';
                }
            });

            tableHtml += '</tbody></table></div>';
            return tableHtml;
        });

        // Math Notation (basic: inline $...$ and block $$...$$)
        html = html.replace(/\$\$(.*?)\$\$/gs, '<div class="math-block text-center my-4 p-3 bg-gray-800 rounded-md overflow-x-auto text-yellow-300 text-sm">$1</div>');
        html = html.replace(/\$(.*?)\$/g, '<span class="math-inline bg-gray-800 text-yellow-300 px-1 py-0.5 rounded-sm text-sm">$1</span>');

        // Inline code
        html = html.replace(/`(.*?)`/g, '<code class="bg-gray-700 text-gray-100 px-1 py-0.5 rounded-sm text-sm font-mono">$1</code>');

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
            const codeHtml = `<div class="code-block-container relative"><pre id="code-block-wrapper" class="line-numbers text-sm p-5 overflow-x-auto"><code id="code-content" class="${languageClass}">${block.code}</code></pre></div>`;
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