function renderFinalResultBubble(result) {
    const container = document.getElementById('final-result-bubble');
    container.innerHTML = `
        <div class="message-bubble fade-in-up p-4 rounded-lg shadow-md mb-4 relative" style="max-width:80%;">
            <div class="sender-label font-bold mb-1 text-gray-300">All Agents (Synthesized):</div>
            <div class="final-response-content">${renderMarkdown(result)}</div>
            <div class="action-buttons flex mt-4 space-x-3 ml-auto pr-2">
                <button class="action-button good-response-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1"><i class="fas fa-thumbs-up"></i><span>Good Response</span></button>
                <button class="action-button bad-response-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1"><i class="fas fa-thumbs-down"></i><span>Bad Response</span></button>
                <button class="action-button read-aloud-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1"><i class="fas fa-volume-up"></i><span>Read Aloud</span></button>
                <button class="action-button edit-canvas-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1"><i class="fas fa-edit"></i><span>Edit Canvas</span></button>
                <button class="action-button share-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1"><i class="fas fa-share-alt"></i><span>Share</span></button>
            </div>
        </div>
    `;
}

// Advanced Markdown to HTML converter with Tailwind classes
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
    // Wrap consecutive <li> elements into a single <ul> with list-style: none
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
        const codeHtml = `<pre id="code-block-wrapper" class="line-numbers text-sm p-5 overflow-x-auto"><code id="code-content" class="${languageClass}">${block.code}</code></pre>`;
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

let finalResultText = '';
let submitButton = null;
let bubbleCreated = false;
function renderFinalResultBubbleStreaming(chunk, done = false) {
    const container = document.getElementById('final-result-bubble');
    // Only create the bubble when the first chunk arrives
    if (!bubbleCreated && chunk) {
        container.innerHTML = `
            <div class="message-bubble fade-in-up p-4 rounded-lg shadow-md mb-4 relative" style="max-width:80%;">
                <div class="sender-label font-bold mb-1 text-gray-300">All Agents (Synthesized):</div>
                <div class="final-response-content" id="final-response-content"></div>
                <div class="action-buttons flex mt-4 space-x-3 ml-auto pr-2" id="final-action-buttons" style="display:none;">
                    <button class="action-button good-response-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1"><i class="fas fa-thumbs-up"></i><span>Good Response</span></button>
                    <button class="action-button bad-response-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1"><i class="fas fa-thumbs-down"></i><span>Bad Response</span></button>
                    <button class="action-button read-aloud-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1"><i class="fas fa-volume-up"></i><span>Read Aloud</span></button>
                    <button class="action-button edit-canvas-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1"><i class="fas fa-edit"></i><span>Edit Canvas</span></button>
                    <button class="action-button share-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1"><i class="fas fa-share-alt"></i><span>Share</span></button>
                </div>
            </div>
        `;
        bubbleCreated = true;
    }
    if (chunk) {
        finalResultText += chunk;
        document.getElementById('final-response-content').innerHTML = renderMarkdown(finalResultText);
        if (window.Prism) Prism.highlightAll();
    }
    if (done) {
        document.getElementById('final-action-buttons').style.display = '';
        // Re-enable submit button and restore color
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.classList.remove('bg-gray-400', 'cursor-not-allowed');
            submitButton.classList.add('hover:bg-blue-600');
        }
        bubbleCreated = false; // Reset for next query
    }
}
function startMultiAgentStream(query) {
    fetch('/multi-agent-stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query})
    }).then(response => {
        const reader = response.body.getReader();
        let buffer = '';
        finalResultText = '';
        function read() {
            reader.read().then(({done, value}) => {
                if (done) return;
                buffer += new TextDecoder().decode(value);
                let lines = buffer.split('\n');
                buffer = lines.pop(); // last line may be incomplete
                for (const line of lines) {
                    if (!line.trim()) continue;
                    const msg = JSON.parse(line);
                    if (msg.type === 'status') {
                        if (!agentStatus[msg.agent]) agentStatus[msg.agent] = {status: msg.status, progress: 0};
                        agentStatus[msg.agent].status = msg.status;
                        if (msg.status === "PROCESSING") {
                            // Start animation if not already running
                            if (typeof msg.progress === 'number' && msg.progress > 0) {
                                agentStatus[msg.agent].progress = msg.progress;
                            }
                            startAgentProgress(msg.agent);
                        } else if (msg.status === "COMPLETED" || msg.status === "FAILED") {
                            // Set to 100% and stop animation
                            agentStatus[msg.agent].progress = 1.0;
                            stopAgentProgress(msg.agent);
                        } else if (msg.status === "PENDING") {
                            agentStatus[msg.agent].progress = 0;
                            stopAgentProgress(msg.agent);
                        }
                        renderAgents();
                        updateThinkingIndicator(); // Update thinking indicator after status change
                    } else if (msg.type === 'final_chunk') {
                        renderFinalResultBubbleStreaming(msg.chunk, false);
                    } else if (msg.type === 'final_done') {
                        renderFinalResultBubbleStreaming('', true);
                    }
                }
                read();
            });
        }
        read();
    });
}