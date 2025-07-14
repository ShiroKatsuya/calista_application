// Main JavaScript for Calista interface - jQuery version

$(document).ready(function() {
    // Initialize the application
    init();
});

function init() {
    // Set up event listeners
    setupEventListeners();
    
    // Initialize tooltips
    initTooltips();
    
    // Set up navigation
    setupNavigation();
}

function setupEventListeners() {
    // Search form submission
    const $searchForm = $('#searchForm');
    if ($searchForm.length) {
        // Remove the handleSearch event listener to avoid duplicate submissions
        $searchForm.off('submit', handleSearch);
    }
    
    // Follow-up form submission
    $('#followUpForm').on('submit', handleFollowUp);
    
    // Action buttons
    $('.action-btn, .action-btn-new, .action-btn-vibrant, .action-btn-gray').on('click', handleAction);
    
    // Navigation items
    $('.nav-item, .nav-item-expandable').on('click', handleNavigation);
    
    // Sidebar toggle
    $('#sidebarToggle').on('click', toggleSidebar);
    
    // Tab navigation
    $('.tab-btn').on('click', handleTabSwitch);
    
    // Search input auto-focus
    $('#searchInput').on('keydown', handleKeyDown);
    
    // Add functionality to search interface buttons
    $('button[title="Attach file"]').on('click', () => {
        showToast('File attachment feature coming soon', 'success');
    });
    
    $('button[title="Voice input"]').on('click', () => {
        showToast('Voice input feature coming soon', 'success');
    });
    
    $('button[title="Upload image"]').on('click', () => {
        showToast('Image upload feature coming soon', 'success');
    });
}

function handleSearch(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const query = formData.get('query');
    
    if (!query.trim()) {
        showToast('Please enter a search query', 'error');
        return;
    }
    
    // Show loading state
    showLoading();
    
    // Send search request

}

function handleAction(e) {
    const action = $(e.currentTarget).data('action');
    const query = $('#searchInput').val().trim();
    
    // Add visual feedback
    $(e.currentTarget).addClass('scale-95');
    setTimeout(() => $(e.currentTarget).removeClass('scale-95'), 150);
    
    if (!query) {
        showToast('Please enter a query first', 'error');
        $('#searchInput').focus();
        return;
    }
    
    // Show loading state
    showLoading();
    
    // Send action request
    const formData = new FormData();
    formData.append('query', query);
    

}

function handleNavigation(e) {
    // Add visual feedback
    $(e.currentTarget).addClass('scale-95');
    setTimeout(() => $(e.currentTarget).removeClass('scale-95'), 150);
    
    // Remove active class from all nav items
    $('.nav-item, .nav-item-expandable').removeClass('active');
    
    // Add active class to clicked item
    $(e.currentTarget).addClass('active');
    
    // Get the navigation action from data attribute or icon
    const navAction = $(e.currentTarget).data('nav');
    let action = navAction || '';
    
    if (!action) {
        // Fallback to icon detection
        const $icon = $(e.currentTarget).find('i');
        if ($icon.hasClass('fa-home')) {
            action = 'home';
        } else if ($icon.hasClass('fa-compass')) {
            action = 'discover';
        } else if ($icon.hasClass('fa-book') || $icon.hasClass('fa-cube')) {
            action = 'spaces';
        } else if ($icon.hasClass('fa-plus')) {
            action = 'new-chat';
        } else if ($icon.hasClass('fa-crown') || $icon.hasClass('fa-arrow-up')) {
            action = 'upgrade';
        } else if ($icon.hasClass('fa-download')) {
            action = 'install';
        } else {
            action = 'account';
        }
    }
    
    // Handle specific navigation actions
    switch(action) {
        case 'new-chat':
            // Reset to initial state
            const $initialState = $('#initialState');
            const $resultsState = $('#resultsState');
            if ($resultsState.length && !$resultsState.hasClass('hidden')) {
                $resultsState.addClass('hidden');
                $initialState.removeClass('hidden');
                $('#searchInput').val('').focus();
            }
            showToast('Started new chat', 'success');
            break;
        case 'home':
            showToast('Welcome home', 'success');
            break;
        case 'discover':
            showToast('Explore new topics', 'success');
            break;
        case 'spaces':
            showToast('Manage your spaces', 'success');
            break;
        case 'account':
            showToast('Account settings', 'success');
            break;
        case 'upgrade':
            showToast('Upgrade to Pro', 'success');
            break;
        case 'install':
            showToast('Install desktop app', 'success');
            break;
        default:
            showToast(`Navigated to ${action}`, 'success');
    }
}

function toggleSidebar() {
    const $sidebar = $('#sidebar');
    const $sidebarToggle = $('#sidebarToggle');
    
    if ($sidebar.hasClass('sidebar-expanded')) {
        // Collapse sidebar
        $sidebar.removeClass('sidebar-expanded').addClass('sidebar-collapsed');
        $sidebarToggle.find('i').removeClass('fa-times').addClass('fa-bars');
    } else {
        // Expand sidebar
        $sidebar.removeClass('sidebar-collapsed').addClass('sidebar-expanded');
        $sidebarToggle.find('i').removeClass('fa-bars').addClass('fa-times');
    }
}

function handleKeyDown(e) {
    // Handle Enter key in search input
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        $('#searchForm').trigger('submit');
    }
}

function showResults(data) {
    // Switch to results layout
    switchToResultsLayout(data.query, data.message);
}

function showLoading() {
    // Check if we're in initial state or results state
    const $initialState = $('#initialState');
    const $resultsState = $('#resultsState');
    
    if (!$initialState.hasClass('hidden')) {
        // We're in initial state, just show a loading message
        const $button = $('#searchForm button[type="submit"]');
        const originalContent = $button.html();
        $button.html('<div class="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>')
               .prop('disabled', true)
               .data('originalContent', originalContent);
    } else {
        // We're in results state, show loading in response area
        $('#responseText').html(`
            <div class="flex items-center">
                <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-primary mr-3"></div>
                <span class="text-gray-300">Processing...</span>
            </div>
        `);
    }
}

function hideLoading() {
    // Restore search button if we modified it
    const $button = $('#searchForm button[type="submit"]');
    if ($button.length && $button.data('originalContent')) {
        $button.html($button.data('originalContent'))
               .prop('disabled', false)
               .removeData('originalContent');
    }
}

function showToast(message, type = 'success') {
    const $toastContainer = $('#toastContainer');
    if (!$toastContainer.length) return;
    
    const icon = type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle';
    const $toast = $(`
        <div class="toast ${type} translate-x-full opacity-0">
            <div class="flex items-center space-x-2">
                <i class="fas ${icon}"></i>
                <span>${message}</span>
            </div>
        </div>
    `);
    
    $toastContainer.append($toast);
    
    // Trigger animation
    setTimeout(() => {
        $toast.addClass('show').removeClass('translate-x-full opacity-0');
    }, 100);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        $toast.addClass('hide').removeClass('show');
        setTimeout(() => {
            $toast.remove();
        }, 300);
    }, 3000);
}

function initTooltips() {
    // Tooltips are handled via CSS :hover pseudo-class
    // This function can be extended for more complex tooltip behavior
}

function setupNavigation() {
    // Additional navigation setup if needed
    // For now, the basic click handlers are sufficient
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function switchToResultsLayout(query, response) {
    const $initialState = $('#initialState');
    const $resultsState = $('#resultsState');
    
    // Set the query and response
    $('#queryTitle').text(query);
    
    // Animate the response text with typing effect
    typeWriterResponse($('#responseText'), response);
    
    // Generate related questions
    generateRelatedQuestions(query);
    
    // Switch layouts with animation
    $initialState.addClass('state-hidden');
    setTimeout(() => {
        $initialState.addClass('hidden');
        $resultsState.removeClass('hidden').addClass('fade-in-up');
        setTimeout(() => {
            $resultsState.addClass('state-visible');
        }, 50);
    }, 500);
}

function typeWriterResponse($element, text) {
    $element.html('').addClass('typing-animation');
    let i = 0;
    
    function type() {
        if (i < text.length) {
            $element.html($element.html() + text.charAt(i));
            i++;
            setTimeout(type, 20);
        } else {
            $element.removeClass('typing-animation');
        }
    }
    
    setTimeout(type, 300);
}

function generateRelatedQuestions(originalQuery) {
    const $relatedContainer = $('#relatedQuestions');
    const questions = [
        "What makes a friendly greeting like \"hi\" more engaging in conversations",
        "How can I respond to your \"hi\" to start a meaningful chat",
        "Why do people often use simple greetings before sharing more details",
        "What are some creative ways to say \"hi\" that catch attention",
        "How does the tone of a \"hi\" influence the mood of our interaction"
    ];
    
    $relatedContainer.empty();
    questions.forEach((question, index) => {
        const $questionEl = $(`
            <div class="related-question">
                <span>${question}</span>
                <i class="fas fa-plus expand-icon"></i>
            </div>
        `);
        $questionEl.on('click', () => handleRelatedQuestion(question));
        
        // Stagger the animation
        setTimeout(() => {
            $relatedContainer.append($questionEl);
            $questionEl.addClass('fade-in-up');
        }, index * 100);
    });
}

function handleRelatedQuestion(question) {
    $('#followUpInput').val(question);
    showToast('Question added to input', 'success');
}

function handleFollowUp(e) {
    e.preventDefault();
    const query = $('#followUpInput').val().trim();
    
    if (!query) {
        showToast('Please enter a follow-up question', 'error');
        return;
    }
    
    // Show loading in the response area
    $('#responseText').html('<div class="flex items-center"><div class="animate-spin rounded-full h-6 w-6 border-b-2 border-primary mr-3"></div>Processing your follow-up...</div>');
    
    // Log the data being sent to /search
    console.log('Sending query to /search:', query);
    

}

function handleTabSwitch(e) {
    $('.tab-btn').removeClass('active');
    $(e.currentTarget).addClass('active');
    
    const tabType = $(e.currentTarget).data('tab');
    showToast(`Switched to ${tabType} tab`, 'success');
}

// Add some extra dynamic effects
function addDynamicEffects() {
    // Add mouse trail effect
    $(document).on('mousemove', (e) => {
        if (Math.random() < 0.1) { // 10% chance to create a particle
            createMouseParticle(e.clientX, e.clientY);
        }
    });
    
    // Add typing animation to search placeholder
    const $searchInput = $('#searchInput');
    if ($searchInput.length) {
        const originalPlaceholder = $searchInput.attr('placeholder');
        let isTyping = false;
        
        $searchInput.on('focus', function() {
            if (!isTyping && !$(this).val()) {
                typeWriterEffect(this, 'What would you like to know?');
            }
        });
        
        $searchInput.on('blur', function() {
            if (!$(this).val()) {
                $(this).attr('placeholder', originalPlaceholder);
            }
        });
    }
}

function createMouseParticle(x, y) {
    const $particle = $('<div class="fixed w-1 h-1 bg-primary/50 rounded-full pointer-events-none z-50"></div>');
    $particle.css({
        left: x + 'px',
        top: y + 'px',
        transform: 'translate(-50%, -50%)'
    });
    
    $('body').append($particle);
    
    // Animate the particle
    $particle.animate({
        top: (y - 150) + 'px',
        opacity: 0
    }, {
        duration: 1000,
        easing: 'swing',
        complete: function() {
            $particle.remove();
        }
    });
}

function typeWriterEffect(element, text) {
    $(element).attr('placeholder', '');
    let i = 0;
    
    function type() {
        if (i < text.length) {
            $(element).attr('placeholder', $(element).attr('placeholder') + text.charAt(i));
            i++;
            setTimeout(type, 100);
        }
    }
    
    type();
}

// Initialize dynamic effects
$(document).ready(() => {
    setTimeout(addDynamicEffects, 1000); // Add effects after initial load
});

// Export functions for potential use in other scripts
window.CalistaApp = {
    showToast,
    showResults,
    showLoading,
    hideLoading,
    createMouseParticle
};