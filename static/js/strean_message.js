let conversationMessages = [];

let userIsScrolling = false; // Flag to track user scroll activity
let scrollTimeout; // To store the timeout for resetting userIsScrolling

const SCROLL_DEBOUNCE_TIME = 500; // Milliseconds to wait before resuming auto-scroll

// Function to handle scroll events
function handleScrollActivity() {
    userIsScrolling = true;
    clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(() => {
        userIsScrolling = false;
    }, SCROLL_DEBOUNCE_TIME);
}

// Initialize scroll listeners once the DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const responseContainerWrapper = document.getElementById('responseContainerWrapper');
    if (responseContainerWrapper) {
        responseContainerWrapper.addEventListener('scroll', handleScrollActivity);
        responseContainerWrapper.addEventListener('wheel', handleScrollActivity); // Also detect wheel for better responsiveness
    }
});

function renderFinalResultBubble(result) {
    const conversationHistory = document.getElementById('conversation-history');
    const resultDiv = document.createElement('div');
    resultDiv.className = "message-bubble fade-in-up p-4 rounded-lg shadow-md mb-4 relative";
    resultDiv.innerHTML = `
        <div class="sender-label font-bold mb-1 text-gray-300">All Agents (Synthesized):</div>
        <div class="final-response-content">${renderMarkdown(result)}</div>
        <div class="action-buttons flex mt-4 space-x-3 ml-auto pr-2">
            <button class="action-button good-response-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1" title="Good Response"><i class="fas fa-thumbs-up"></i><span>Good Response</span></button>
            <button class="action-button bad-response-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1" title="Bad Response"><i class="fas fa-thumbs-down"></i><span>Bad Response</span></button>
            <button class="action-button read-aloud-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1" title="Read Aloud"><i class="fas fa-volume-up"></i><span>Read Aloud</span></button>
            <button class="action-button edit-canvas-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1" title="Edit Canvas"><i class="fas fa-edit"></i><span>Edit Canvas</span></button>
            <button class="action-button share-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1" title="Share"><i class="fas fa-share-alt"></i><span>Share</span></button>
        </div>
    `;
    conversationHistory.appendChild(resultDiv);
    addConversationMessage('all_agents', result, 'final_result');
    scrollToBottom();
}





// New helper function to render markdown *inside* the <think> blocks
// This function will *not* look for <think> tags itself, and will assume its input is already raw text that needs escaping and markdown processing.
function _renderThinkContentMarkdown(rawText) {
    let html = escapeHtml(rawText); // Escape the raw text first

    // Store code blocks and replace with placeholders
    const codeBlocks = [];
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        const placeholder = `CODE_BLOCK_PLACEHOLDER_${codeBlocks.length}__`;
        codeBlocks.push({ lang: lang, code: code });
        return placeholder;
    });

    // Horizontal Rule
    html = html.replace(/^[\s\*\-_]{3,}\s*$/gm, '<hr class="my-2 border-t-1 border-gray-500">');

    // Headers (H1-H6) - use smaller sizes for nested content
    html = html.replace(/^###### (.*)$/gm, '<h6 class="text-xs font-semibold mt-2 mb-1 text-gray-300">$1</h6>');
    html = html.replace(/^##### (.*)$/gm, '<h5 class="text-sm font-semibold mt-2 mb-1 text-gray-300">$1</h5>');
    html = html.replace(/^#### (.*)$/gm, '<h4 class="text-base font-semibold mt-2 mb-1 text-gray-300">$1</h4>');
    html = html.replace(/^### (.*)$/gm, '<h3 class="text-lg font-semibold mt-2 mb-2 text-gray-200">$1</h3>');
    html = html.replace(/^## (.*)$/gm, '<h2 class="text-xl font-bold mt-3 mb-2 text-gray-100">$1</h2>');
    html = html.replace(/^# (.*)$/gm, '<h1 class="text-2xl font-bold mt-4 mb-3 text-white">$1</h1>');

    // Blockquotes
    html = html.replace(/^> (.*)$/gm, '<blockquote class="border-l-3 border-blue-400 pl-3 py-0.5 italic text-gray-300 my-2">$1</blockquote>');

    // Lists (unordered and ordered)
    html = html.replace(/\n(?= *- |^\d+\. )/g, '@@NEWLINE_HOLDER_INNER@@');
    // Unordered lists
    html = html.replace(/^- (.*)$/gm, '<li class="mb-0.5">$1</li>');
    // Ordered lists
    html = html.replace(/^(\d+)\. (.*)$/gm, '<li class="mb-0.5">$1. $2</li>');
    // Wrap consecutive <li> elements into a single <ul> with list-style: none
    html = html.replace(/((?:<li class="mb-0.5">.*?<\/li>\n?)+)/g, (match, listItems) => {
        listItems = listItems.trim();
        return `<ul class="list-none pl-4 mb-1 text-gray-200">${listItems}</ul>`;
    });
    html = html.replace(/@@NEWLINE_HOLDER_INNER@@/g, '<br>');

    // Links
    html = html.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-blue-300 hover:underline">$1</a>');

    // Images (less likely in thought blocks, but good to support)
    html = html.replace(/!\[(.*?)\]\((.*?)\)/g, '<img src="$2" alt="$1" class="max-w-full h-auto rounded my-2 shadow">');

    // Math Notation (basic: inline $...$ and block $$...$$)
    html = html.replace(/\$\$(.*?)\$\$/gs, '<div class="math-block text-center my-2 p-2 bg-gray-700 rounded-md overflow-x-auto text-yellow-300 text-xs">$1</div>');
    html = html.replace(/\$(.*?)\$/g, '<span class="math-inline bg-gray-700 text-yellow-300 px-0.5 py-0 rounded-sm text-xs">$1</span>');

    // Inline code
    html = html.replace(/`(.*?)`/g, '<code class="bg-gray-600 text-gray-100 px-0.5 py-0 rounded-sm text-xs font-mono">$1</code>');

    // Newlines to <br>
    html = html.replace(/\n/g, '<br>');

    // Basic bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Basic italics
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Restore code blocks (for thought content)
    for (let i = 0; i < codeBlocks.length; i++) {
        const placeholder = `CODE_BLOCK_PLACEHOLDER_${i}__`;
        const block = codeBlocks[i];
        const languageClass = block.lang ? `language-${block.lang}` : 'language-markup';
        // Don't escape code content - we want to display the actual characters
        const codeHtml = `<pre class="line-numbers text-xs p-3 overflow-x-auto"><code class="${languageClass}">${block.code}</code></pre>`;
        html = html.replace(placeholder, codeHtml);
    }
    return html;
}

// Advanced Markdown to HTML converter with Tailwind classes
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
        // Don't escape code content - we want to display the actual characters
        const codeHtml = `<pre id="code-block-wrapper" class="line-numbers text-sm p-5 overflow-x-auto"><code id="code-content" class="${languageClass}">${block.code}</code></pre>`;
        html = html.replace(placeholder, codeHtml);
    }

    // 4. Finally, restore the <think> blocks and recursively render their content
    for (let i = 0; i < thinkBlockData.length; i++) {
        const placeholder = `__THINK_BLOCK_${i}__`;
        // Recursively render markdown for the content inside <think>
        const renderedThinkContent = _renderThinkContentMarkdown(thinkBlockData[i]);
        const thinkHtml = `<details class="thought-process"><summary class="thought-process-summary text-gray-400 hover:text-blue-400 cursor-pointer font-semibold mb-2">Thought Process</summary><div class="thought-process-content p-3 bg-gray-700 rounded-md mt-2">${renderedThinkContent}</div></details>`;
        html = html.replace(placeholder, thinkHtml);
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

// Utility function to scroll to the bottom of the response container
function scrollToBottom() {
    const responseContainerWrapper = document.getElementById('responseContainerWrapper');
    if (responseContainerWrapper && !userIsScrolling) { // Only scroll if user is not actively scrolling
        responseContainerWrapper.scrollTo({
            top: responseContainerWrapper.scrollHeight,
            behavior: 'smooth'
        });
    }
}

// Utility function to scroll to a specific element
function scrollToElement(element) {
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'end',
            inline: 'nearest'
        });
    }
}

let finalResultText = '';
let submitButton = null;
let bubbleCreated = false;
function renderFinalResultBubbleStreaming(chunk, done = false) {
    const conversationHistory = document.getElementById('conversation-history');
    if (!bubbleCreated && chunk) {
        streamingResultDiv = document.createElement('div');
        streamingResultDiv.className = "message-bubble fade-in-up p-4 rounded-lg shadow-md mb-4 relative";
        streamingResultDiv.innerHTML = `
            <div class="sender-label font-bold mb-1 text-gray-300">All Agents (Synthesized):</div>
            <div class="final-response-content" id="final-response-content"></div>
            <div class="action-buttons flex mt-4 space-x-3 ml-auto pr-2" id="final-action-buttons" style="display:none;">
                <button class="action-button good-response-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1" title="Good Response"><i class="fas fa-thumbs-up"></i><span>Good Response</span></button>
                <button class="action-button bad-response-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1" title="Bad Response"><i class="fas fa-thumbs-down"></i><span>Bad Response</span></button>
                <button class="action-button read-aloud-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1" title="Read Aloud"><i class="fas fa-volume-up"></i><span>Read Aloud</span></button>
                <button class="action-button edit-canvas-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1" title="Edit Canvas"><i class="fas fa-edit"></i><span>Edit Canvas</span></button>
                <button class="action-button share-button text-gray-400 hover:text-blue-500 transition-colors text-sm flex items-center space-x-1" title="Share"><i class="fas fa-share-alt"></i><span>Share</span></button>
            </div>
        `;
        conversationHistory.appendChild(streamingResultDiv);
        bubbleCreated = true;
        scrollToBottom();
    }
    if (chunk && streamingResultDiv) {
        finalResultText += chunk;
        streamingResultDiv.querySelector('#final-response-content').innerHTML = renderMarkdown(finalResultText);
        if (window.Prism) Prism.highlightAll();
        scrollToBottom();
    }
    if (done && streamingResultDiv) {
        streamingResultDiv.querySelector('#final-action-buttons').style.display = '';
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.classList.remove('bg-gray-400', 'cursor-not-allowed');
            submitButton.classList.add('hover:bg-blue-600');
        }
        bubbleCreated = false;
        addConversationMessage('all_agents', finalResultText, 'final_result');
        streamingResultDiv = null;
        finalResultText = '';
    }
}

function saveConversationToJson() {
    if (!conversationMessages || conversationMessages.length === 0) {
        addSystemMessage('No conversation data to save.');
        return;
    }
    fetch('/save_conversation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ conversation: conversationMessages })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            console.log('Conversation saved successfully:', data.file_path);
            // Optionally, inform the user or provide a download link
        } else {
            console.error('Error saving conversation:', data.message);
            addSystemMessage('Error saving conversation: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Fetch error when saving conversation:', error);
        addSystemMessage('Network error when trying to save conversation.');
    });
}

function loadConversationHistoryAllAgent() {
    fetch('/get_all_conversation/agent')
    .then(response => response.json())
    .then(data => {
        if (Array.isArray(data)) {
            // Update the in-memory conversationMessages array
            conversationMessages = data;
            // Clear the UI and re-render all messages
            const container = document.getElementById('final-result-bubble');
            if (container) container.innerHTML = '';
            data.forEach(msg => {
                if (msg.type === 'HumanMessage' || msg.type === 'user') {
                    addMessage(msg.content, 'user');
                } else if (msg.type === 'AIMessage' || msg.type === 'ai' || msg.type === 'final_result') {
                    addMessage(msg.content, msg.sender || 'ai');
                }
            });
        }
    })
    .catch(error => {
        console.error('Error loading conversation history:', error);
    });
}


function startMultiAgentStream(query) {
    // Add the new user message to the conversationMessages array before sending
    addConversationMessage('user', query, 'HumanMessage');

    fetch('/multi-agent-stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            query,
            messages: conversationMessages // send the full history
        })
    })
    .then(response => {
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
                        saveConversationToJson();
                    }
                }
                read();
            });
        }
        read();
    });
}

// Add a helper to add messages to the conversation history
function addConversationMessage(sender, content, type = 'chat') {
    conversationMessages.push({ sender, content, type, timestamp: new Date().toISOString() });
}