let isChatStreamActive = false;
let conversationMessages = []; // Array to store all conversation messages
let currentStreamingMessage = null; // Track the current streaming message

// Utility function to scroll to the bottom of the response container
function scrollToBottom() {
    const responseContainerWrapper = document.getElementById('responseContainerWrapper');
    if (responseContainerWrapper) {
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

$(document).ready(function() {
    const $searchForm = $('#searchForm');
    const $searchInput = $('#searchInput'); // Renamed from $queryInput in some contexts
    const $initialState = $('#initialState');
    const $resultsState = $('#resultsState');
    const $responseContainer = $('#responseContainer');
    const $followUpForm = $('#followUpForm');
    const $followUpInput = $('#followUpInput');
    const $loadingIndicator = $('#loadingIndicator');
    const $thinkingIndicator = $('#thinkingIndicator');
    const $clearInputButton = $('#clearInputButton'); // For the main input clear button

    // Function to add a chunk to an existing streaming message or create a new one
    function addMessageChunk(text, sender) {
        // Suppress any message containing ROUTE_TO:
        if (typeof text === 'string' && text.includes('ROUTE_TO:')) {
            return;
        }
        console.log(`Adding chunk: [${sender}] ${text.substring(0, 50)}...`);
        
        // Check if we have an existing streaming message for this sender
        if (currentStreamingMessage && currentStreamingMessage.sender === sender) {
            // Add to existing message
            currentStreamingMessage.content += text;
            currentStreamingMessage.$messageContent.html(renderMarkdown(currentStreamingMessage.content));
            
            // Scroll to the bottom
            scrollToBottom();
            
            // Re-highlight code blocks
            Prism.highlightAllUnder(currentStreamingMessage.$messageDiv[0]);
            
            console.log(`Updated streaming message for ${sender}, total length: ${currentStreamingMessage.content.length}`);
        } else {
            // Create new streaming message
            console.log(`Creating new streaming message for ${sender}`);
            const $messageDiv = $('<div>').addClass('message-bubble fade-in-up p-4 rounded-lg shadow-md mb-4 relative');

            // Apply sender-specific Tailwind classes and margin
            if (sender === 'user') {
                $messageDiv.addClass('bg-blue-600 text-white ml-auto rounded-br-none').css('max-width', '80%');
            // } else if (sender == 'ai') {
            //     $messageDiv.addClass('bg-gray-700 text-gray-100 mr-auto rounded-bl-none').css('max-width', '80%');
            } else if (sender === 'supervisor') {
                $messageDiv.addClass('bg-purple-800 text-white mr-auto rounded-bl-none').css('max-width', '80%');
            } else if (sender === 'system-error') {
                $messageDiv.addClass('bg-red-800 text-white mr-auto rounded-bl-none').css('max-width', '80%');
            } else if (sender === 'Rina') {
                $messageDiv.addClass(' text-white mr-auto rounded-bl-none').css('max-width', '80%');
            } else if (sender === 'Emilia') {
                $messageDiv.addClass(' text-white mr-auto rounded-bl-none').css('max-width', '80%');
            }

            // Add sender label for AI/agent messages
            if (sender !== 'user') {
                const $senderLabel = $('<div>').addClass('sender-label font-bold mb-1 text-gray-300');
                $senderLabel.text(sender.charAt(0).toUpperCase() + sender.slice(1) + ':');
                $messageDiv.append($senderLabel);
            }

            const $messageContent = $('<div>').addClass('message-content prose prose-invert max-w-none');
            $messageContent.html(renderMarkdown(text));

            $messageDiv.append($messageContent);

            // Append to the main response container
            $responseContainer.append($messageDiv);

            // Scroll to the bottom of the conversation
            scrollToBottom();
            
            // Add copy buttons to any new code blocks
            addCopyButtons($messageDiv);

            // After adding a new message with potentially new code blocks, highlight them
            Prism.highlightAllUnder($messageDiv[0]);

            // Add action buttons to all messages
            const $actionButtonsContainer = createActionButtons();
            $messageDiv.append($actionButtonsContainer);

            // Store reference to current streaming message
            currentStreamingMessage = {
                sender: sender,
                content: text,
                $messageDiv: $messageDiv,
                $messageContent: $messageContent
            };
        }
    }

    // Function to add a regular message to the conversation
    function addMessage(text, sender) {
        // Suppress any message containing ROUTE_TO:
        if (typeof text === 'string' && text.includes('ROUTE_TO:')) {
            return;
        }
        console.log(`Adding message: [${sender}] ${text.substring(0, 50)}...`);
        
        // Clear current streaming message if it's from the same sender
        if (currentStreamingMessage && currentStreamingMessage.sender === sender) {
            console.log(`Replacing streaming message for ${sender} with complete message`);
            // Remove the streaming message from DOM
            currentStreamingMessage.$messageDiv.remove();
            currentStreamingMessage = null;
        }
        
        const $messageDiv = $('<div>').addClass('message-bubble fade-in-up p-4 rounded-lg shadow-md mb-4 relative');

        // Apply sender-specific Tailwind classes and margin
        if (sender === 'user') {
            $messageDiv.addClass('bg-blue-600 text-white ml-auto rounded-br-none').css('max-width', '80%');
        // } else if (sender == 'ai') {
        //     $messageDiv.addClass('bg-gray-700 text-gray-100 mr-auto rounded-bl-none').css('max-width', '80%');
    
        } else if (sender === 'supervisor') {
            $messageDiv.addClass('bg-purple-800 text-white mr-auto rounded-bl-none').css('max-width', '80%');
        } else if (sender === 'system-error') {
            $messageDiv.addClass('bg-red-800 text-white mr-auto rounded-bl-none').css('max-width', '80%');
        } else if (sender === 'Rina') {
            $messageDiv.addClass(' text-white mr-auto rounded-bl-none').css('max-width', '80%');
        } else if (sender === 'Emilia') {
            $messageDiv.addClass(' text-white mr-auto rounded-bl-none').css('max-width', '80%');
        }

        // Add sender label for AI/agent messages
        if (sender !== 'user') {
            const $senderLabel = $('<div>').addClass('sender-label font-bold mb-1 text-gray-300');
            $senderLabel.text(sender.charAt(0).toUpperCase() + sender.slice(1) + ':');
            $messageDiv.append($senderLabel);
        }

        // Replace \n with <br> for newlines
        const formattedText = renderMarkdown(text);

        const $messageContent = $('<div>').addClass('message-content prose prose-invert max-w-none');
        $messageContent.html(formattedText);

        $messageDiv.append($messageContent);

        // Append to the main response container
        $responseContainer.append($messageDiv);

        // Scroll to the bottom of the conversation
        scrollToBottom();
        
        // Add copy buttons to any new code blocks
        addCopyButtons($messageDiv);

        // After adding a new message with potentially new code blocks, highlight them
        Prism.highlightAllUnder($messageDiv[0]);

        // Add action buttons to all messages
        const $actionButtonsContainer = createActionButtons();
        $messageDiv.append($actionButtonsContainer);

        console.log("Message Div InnerHTML after buttons:", $messageDiv.html());
        conversationMessages.push({ type: 'chat', sender: sender, content: text }); // Store chat message
    }

    // Function to add a system message (like routing or start/end signals)
    function addSystemMessage(text, sender = 'system') {
        // Remove any existing system-message divs to ensure only one is shown
        $responseContainer.find('.system-message').remove();

        const $systemMessageDiv = $('<div>').addClass('system-message text-center text-gray-400 text-sm my-3');
        $systemMessageDiv.text(text);
        $responseContainer.append($systemMessageDiv);
        scrollToBottom();
        // Save as a chat message with sender, not just as type system
        conversationMessages.push({ type: 'chat', sender: sender, content: text });
    }

    // Clear input button logic
    function toggleClearButton() {
        if ($searchInput.val().trim() !== '' || $followUpInput.val().trim() !== '') {
            $clearInputButton.show();
        } else {
            $clearInputButton.hide();
        }
    }

    // Function to save conversation to JSON
    function saveConversationToJson() {
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

    // Function to load existing conversation history
    function loadConversationHistory() {
        fetch('/get_conversation_history')
        .then(response => response.json())
        .then(data => {
            if (data.conversation_messages && data.conversation_messages.length > 0) {
                // Show results state
                $initialState.addClass('hidden');
                $resultsState.removeClass('hidden');
                
                // Set conversation title
                const $conversationTitle = $('#conversationTitle');
                if ($conversationTitle.length && data.conversation_title) {
                    $conversationTitle.text(data.conversation_title);
                }
                
                // Display existing messages
                data.conversation_messages.forEach(msg => {
                    if (msg.type === 'HumanMessage') {
                        addMessage(msg.content, 'user');
                    } else if (msg.type === 'AIMessage') {
                        addMessage(msg.content, msg.name || 'ai');
                    }
                });
            }
        })
        .catch(error => {
            console.error('Error loading conversation history:', error);
        });
    }

    if ($searchForm.length && $searchInput.length && $initialState.length && $resultsState.length && $responseContainer.length && $followUpForm.length && $followUpInput.length) {
        // Load existing conversation history on page load
        loadConversationHistory();
        
        // Event listener for sending the main query
        $searchForm.on('submit', function(e) {
            e.preventDefault();
            const query = $searchInput.val().trim();

            if (!query) {
                return;
            }

            // Prevent multiple concurrent requests
            if (isChatStreamActive) {
                console.warn('A chat stream is already active. Ignoring new request.');
                return;
            }
            isChatStreamActive = true;

            // Show results state and hide initial state
            $initialState.addClass('hidden');
            $resultsState.removeClass('hidden');

            // Clear previous messages from responseContainer
            $responseContainer.empty();
            
            // Reset streaming message state
            currentStreamingMessage = null;

            // Set the header title to the user's query
            const $conversationTitle = $('#conversationTitle');
            if ($conversationTitle.length) {
                $conversationTitle.text(query);
            }

            // Add the initial user message
            addMessage(query, 'user');

            // Show loading indicator
            $loadingIndicator.removeClass('hidden');

            // Show thinking indicator
            $thinkingIndicator.removeClass('hidden');

            // Send query to backend and stream response
            fetch('/chat_stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query, is_new_conversation: true })
            })
            .then(response => {
                // Hide loading indicator once response starts
                $loadingIndicator.addClass('hidden');

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                function read() {
                    reader.read().then(({ done, value }) => {
                        if (done) {
                            console.log('Stream complete');
                            // Finalize any streaming message
                            if (currentStreamingMessage) {
                                // Convert streaming message to regular message
                                addMessage(currentStreamingMessage.content, currentStreamingMessage.sender);
                                currentStreamingMessage = null;
                            }
                            addSystemMessage('--- Conversation Complete ---');
                            isChatStreamActive = false; // Reset flag on stream completion
                            saveConversationToJson(); // Save conversation
                            return;
                        }

                        buffer += decoder.decode(value, { stream: true });
                        let lines = buffer.split('\n\n'); // SSE messages are separated by double newlines
                        buffer = lines.pop(); // Keep the last (potentially incomplete) line in buffer

                        console.log('Received lines from stream:', lines);

                        // Hide thinking indicator once first data is received
                        if (lines.some(line => line.startsWith('data: '))) {
                            $thinkingIndicator.addClass('hidden');
                        }

                        lines.forEach(line => {
                            if (line.startsWith('data: ')) {
                                try {
                                    const jsonData = JSON.parse(line.substring(6));
                                    console.log('Parsed JSON data:', jsonData);
                                    
                                    // Handle different message types
                                    if (jsonData.type === 'chunk') {
                                        // Add chunk to existing message or create new one
                                        addMessageChunk(jsonData.content, jsonData.sender);
                                    } else if (jsonData.type === 'complete') {
                                        // Add complete message (this will replace any streaming message)
                                        addMessage(jsonData.content, jsonData.sender);
                                    } else {
                                        // Legacy format - add as complete message
                                        addMessage(jsonData.content, jsonData.sender);
                                    }

                                    // Check for routing messages within the content
                                    if (jsonData.content.includes("ROUTE_TO:")) {
                                        const routeMatch = jsonData.content.match(/ROUTE_TO: (\w+) - (.*)/);
                                        if (routeMatch && routeMatch[1] && routeMatch[2]) {
                                            if (routeMatch[1] === "FINISH") {
                                                addSystemMessage(`--- Supervisor: ${routeMatch[2]} ---`, 'supervisor');
                                            } else {
                                                // Use the routeMatch[1] as sender (e.e.g., Rina, Emilia, etc.)
                                                addSystemMessage(`--- Supervisor routing to: ${routeMatch[1]} - ${routeMatch[2]} ---`, routeMatch[1].toLowerCase());
                                            }
                                        }
                                    }

                                } catch (e) {
                                    console.error('Error parsing JSON from stream:', e, 'Line:', line);
                                    // Fallback for non-JSON data, append as plain text to a generic AI message
                                    if (line.substring(6).includes('ROUTE_TO:')) {
                                        return;
                                    }
                                    addMessage(line.substring(6), 'ai');
                                }
                            } else if (line.startsWith('event: end_stream')) {
                                console.log('Event: end_stream received. Finalizing AI response.');
                                // The "--- Conversation Complete ---" message is handled when `done` is true
                            }
                        });
                        read(); // Continue reading
                    }).catch(error => {
                        console.error('Stream reading error:', error);
                        addMessage('Error receiving response.', 'system-error');
                        isChatStreamActive = false; // Reset flag on error
                    });
                }
                read();
            })
            .catch(error => {
                console.error('Fetch error:', error);
                addMessage('Error sending query.', 'system-error');
                isChatStreamActive = false; // Reset flag on error
            });
        });

        // Handle follow-up questions
        $followUpForm.on('submit', function(e) {
            e.preventDefault();
            const followUpQuery = $followUpInput.val().trim();
            if (!followUpQuery) {
                return;
            }

            // Prevent multiple concurrent requests for follow-ups
            if (isChatStreamActive) {
                console.warn('A chat stream is already active. Ignoring new follow-up request.');
                return;
            }
            isChatStreamActive = true;

            addMessage(followUpQuery, 'user');
            $followUpInput.val(''); // Clear follow-up input
            toggleClearButton(); // Hide clear button after submission
            
            // Reset streaming message state for follow-up
            currentStreamingMessage = null;

            // Directly send follow-up query to backend and stream response
            $loadingIndicator.removeClass('hidden'); // Show loading indicator

            // Show thinking indicator for follow-up
            $thinkingIndicator.removeClass('hidden');

            fetch('/chat_stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: followUpQuery, is_new_conversation: false })
            })
            .then(response => {
                $loadingIndicator.addClass('hidden'); // Hide loading indicator once response starts
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                function readFollowUp() {
                    reader.read().then(({ done, value }) => {
                        if (done) {
                            console.log('Follow-up stream complete');
                            // Finalize any streaming message
                            if (currentStreamingMessage) {
                                // Convert streaming message to regular message
                                addMessage(currentStreamingMessage.content, currentStreamingMessage.sender);
                                currentStreamingMessage = null;
                            }
                            addSystemMessage('--- Follow-up Conversation Complete ---');
                            isChatStreamActive = false; // Reset flag on stream completion
                            saveConversationToJson(); // Save conversation after follow-up
                            return;
                        }
                        buffer += decoder.decode(value, { stream: true });
                        let lines = buffer.split('\n\n');
                        buffer = lines.pop();

                        console.log('Received lines from follow-up stream:', lines);

                        // Hide thinking indicator once first data is received for follow-up
                        if (lines.some(line => line.startsWith('data: '))) {
                            $thinkingIndicator.addClass('hidden');
                        }

                        lines.forEach(line => {
                            if (line.startsWith('data: ')) {
                                try {
                                    const jsonData = JSON.parse(line.substring(6));
                                    
                                    // Handle different message types
                                    if (jsonData.type === 'chunk') {
                                        // Add chunk to existing message or create new one
                                        addMessageChunk(jsonData.content, jsonData.sender);
                                    } else if (jsonData.type === 'complete') {
                                        // Add complete message (this will replace any streaming message)
                                        addMessage(jsonData.content, jsonData.sender);
                                    } else {
                                        // Legacy format - add as complete message
                                        addMessage(jsonData.content, jsonData.sender);
                                    }

                                    // Check for routing messages within the content
                                    if (jsonData.content.includes("ROUTE_TO:")) {
                                        const routeMatch = jsonData.content.match(/ROUTE_TO: (\w+) - (.*)/);
                                        if (routeMatch && routeMatch[1] && routeMatch[2]) {
                                            if (routeMatch[1] === "FINISH") {
                                                addSystemMessage(`--- Supervisor: ${routeMatch[2]} ---`, 'supervisor');
                                            } else {
                                                addSystemMessage(`--- Supervisor routing to: ${routeMatch[1]} - ${routeMatch[2]} ---`, routeMatch[1].toLowerCase());
                                            }
                                        }
                                    }
                                } catch (e) {
                                    console.error('Error parsing JSON from follow-up stream:', e, 'Line:', line);
                                    addMessage(line.substring(6), 'ai');
                                }
                            } else if (line.startsWith('event: end_stream')) {
                                console.log('Event: end_stream received for follow-up. Finalizing AI response.');
                            }
                        });
                        readFollowUp();
                    }).catch(error => {
                        console.error('Follow-up stream reading error:', error);
                        addMessage('Error receiving follow-up response.', 'system-error');
                        isChatStreamActive = false; // Reset flag on error
                    });
                }
                readFollowUp();
            })
            .catch(error => {
                console.error('Follow-up fetch error:', error);
                addMessage('Error sending follow-up query.', 'system-error');
                isChatStreamActive = false; // Reset flag on error
            });
        });

        // Handle clicks on related questions
        const $relatedQuestionsContainer = $('#relatedQuestionsContainer');
        if ($relatedQuestionsContainer.length) {
            $relatedQuestionsContainer.on('click', '.related-question', function() {
                const questionText = $(this).find('span').text().trim();
                $followUpInput.val(questionText);
                $followUpForm[0].requestSubmit();
            });
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
            // Convert markdown list items to <li> tags
            html = html.replace(/^- (.*)$/gm, '<li class="mb-1">$1</li>');

            // Ordered lists
            // Convert markdown ordered list items to <li> tags, keeping the number in the text
            html = html.replace(/^(\d+)\. (.*)$/gm, '<li class="mb-1">$1. $2</li>');

            // Wrap consecutive <li> elements into a single <ul> with list-style: none
            // This regex looks for one or more lines that start with <li class="mb-1"> and end with </li>.
            // The \n? allows for a single newline between list items, ensuring blocks are correctly grouped.
            // This handles both unordered and ordered lists by wrapping them in a single <ul> structure.
            html = html.replace(/((?:<li class="mb-1">.*?<\/li>\n?)+)/g, (match, listItems) => {
                // Remove any trailing newlines from the captured listItems block before wrapping
                listItems = listItems.trim();
                // Wrap the list items in a <ul> with the desired classes, including list-none
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

        // Clear input button logic
        $searchInput.on('input', toggleClearButton);
        $clearInputButton.on('click', function() {
            $searchInput.val('');
            toggleClearButton();
            $searchInput.focus();
        });
    }
});