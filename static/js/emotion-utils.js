/**
 * Simple Emotion-like CSS-in-JS utility for dynamic styling
 * Provides theme management and dynamic style generation
 */

class EmotionUtils {
    constructor() {
        this.themes = {
            dark: {
                primary: '#3b82f6',
                secondary: '#60a5fa',
                background: '#111827',
                surface: '#1f2937',
                text: '#f3f4f6',
                textSecondary: '#9ca3af',
                border: '#374151',
                accent: '#fbbf24',
                error: '#ef4444',
                success: '#10b981',
                warning: '#f59e0b'
            },
            light: {
                primary: '#2563eb',
                secondary: '#3b82f6',
                background: '#ffffff',
                surface: '#f9fafb',
                text: '#111827',
                textSecondary: '#6b7280',
                border: '#d1d5db',
                accent: '#f59e0b',
                error: '#dc2626',
                success: '#059669',
                warning: '#d97706'
            }
        };
        
        this.currentTheme = 'dark';
        this.styleSheet = null;
        this.init();
    }

    init() {
        // Create a stylesheet for dynamic styles
        this.createStyleSheet();
        this.applyTheme(this.currentTheme);
    }

    createStyleSheet() {
        if (!this.styleSheet) {
            const style = document.createElement('style');
            style.id = 'emotion-dynamic-styles';
            document.head.appendChild(style);
            this.styleSheet = style.sheet;
        }
    }

    applyTheme(themeName) {
        this.currentTheme = themeName;
        const theme = this.themes[themeName];
        
        // Update CSS custom properties
        const root = document.documentElement;
        Object.entries(theme).forEach(([key, value]) => {
            root.style.setProperty(`--color-${key}`, value);
        });

        // Add theme class to body
        document.body.className = document.body.className.replace(/theme-\w+/g, '');
        document.body.classList.add(`theme-${themeName}`);
    }

    css(strings, ...values) {
        // Simple template literal to CSS converter
        let css = '';
        for (let i = 0; i < strings.length; i++) {
            css += strings[i];
            if (i < values.length) {
                css += values[i];
            }
        }
        return css;
    }

    injectGlobal(css) {
        if (this.styleSheet) {
            try {
                this.styleSheet.insertRule(css, this.styleSheet.cssRules.length);
            } catch (e) {
                console.warn('Failed to inject CSS rule:', e);
            }
        }
    }

    keyframes(name, steps) {
        const keyframeRule = `@keyframes ${name} { ${steps} }`;
        this.injectGlobal(keyframeRule);
        return name;
    }

    createGlobalStyles() {
        const globalStyles = `
            .theme-dark {
                color-scheme: dark;
            }
            
            .theme-light {
                color-scheme: light;
            }
            
            .markdown-container {
                transition: all 0.3s ease;
            }
            
            .markdown-container.theme-dark {
                background-color: var(--color-background);
                color: var(--color-text);
            }
            
            .markdown-container.theme-light {
                background-color: var(--color-background);
                color: var(--color-text);
            }
            
            /* Enhanced animations */
            .markdown-fade-in {
                animation: markdown-fade-in 0.6s ease-out;
            }
            
            .markdown-slide-up {
                animation: markdown-slide-up 0.8s ease-out;
            }
            
            .markdown-bounce-in {
                animation: markdown-bounce-in 0.8s ease-out;
            }
            
            /* Responsive enhancements */
            @media (max-width: 640px) {
                .markdown-container {
                    padding: 1rem;
                }
                
                .markdown-pre {
                    font-size: 0.75rem;
                    padding: 0.75rem;
                }
                
                .markdown-table-wrapper {
                    margin: 0.5rem 0;
                }
            }
            
            /* Print optimizations */
            @media print {
                .markdown-container {
                    color: #000 !important;
                    background: #fff !important;
                }
                
                .markdown-pre {
                    background: #f5f5f5 !important;
                    border: 1px solid #ddd !important;
                }
            }
        `;
        
        this.injectGlobal(globalStyles);
    }

    // Enhanced animation keyframes
    createAnimations() {
        const fadeIn = this.keyframes('markdown-fade-in', `
            0% { opacity: 0; }
            100% { opacity: 1; }
        `);
        
        const slideUp = this.keyframes('markdown-slide-up', `
            0% { opacity: 0; transform: translateY(30px); }
            100% { opacity: 1; transform: translateY(0); }
        `);
        
        const bounceIn = this.keyframes('markdown-bounce-in', `
            0% { opacity: 0; transform: scale(0.3); }
            50% { opacity: 1; transform: scale(1.05); }
            70% { transform: scale(0.9); }
            100% { opacity: 1; transform: scale(1); }
        `);
        
        return { fadeIn, slideUp, bounceIn };
    }

    // Utility for creating responsive styles
    responsive(breakpoints) {
        return Object.entries(breakpoints).map(([breakpoint, styles]) => {
            const mediaQuery = this.getMediaQuery(breakpoint);
            return `@media ${mediaQuery} { ${styles} }`;
        }).join('\n');
    }

    getMediaQuery(breakpoint) {
        const breakpoints = {
            sm: '(min-width: 640px)',
            md: '(min-width: 768px)',
            lg: '(min-width: 1024px)',
            xl: '(min-width: 1280px)',
            '2xl': '(min-width: 1536px)'
        };
        return breakpoints[breakpoint] || breakpoint;
    }

    // Theme-aware color utility
    themeColor(colorKey) {
        return `var(--color-${colorKey})`;
    }

    // Enhanced markdown styling with theme support
    enhanceMarkdownStyles() {
        const enhancedStyles = `
            .markdown-container {
                --markdown-primary: var(--color-primary);
                --markdown-secondary: var(--color-secondary);
                --markdown-background: var(--color-background);
                --markdown-surface: var(--color-surface);
                --markdown-text: var(--color-text);
                --markdown-text-secondary: var(--color-textSecondary);
                --markdown-border: var(--color-border);
                --markdown-accent: var(--color-accent);
            }
            
            .markdown-h1, .markdown-h2, .markdown-h3 {
                color: var(--markdown-text);
                border-bottom-color: var(--markdown-border);
            }
            
            .markdown-blockquote {
                border-left-color: var(--markdown-primary);
                background-color: var(--markdown-surface);
                color: var(--markdown-text-secondary);
            }
            
            .markdown-a {
                color: var(--markdown-primary);
            }
            
            .markdown-a:hover {
                color: var(--markdown-secondary);
            }
            
            .markdown-pre {
                background-color: var(--markdown-surface);
                border-color: var(--markdown-border);
            }
            
            .markdown-code, .markdown-code-inline {
                background-color: var(--markdown-surface);
                border-color: var(--markdown-border);
                color: var(--markdown-text);
            }
            
            .markdown-table {
                background-color: var(--markdown-surface);
            }
            
            .markdown-table thead {
                background-color: var(--markdown-border);
            }
            
            .markdown-table tbody tr:hover {
                background-color: var(--markdown-border);
            }
            
            .markdown-math-block, .markdown-math-inline {
                background-color: var(--markdown-surface);
                border-color: var(--markdown-border);
                color: var(--markdown-accent);
            }
        `;
        
        this.injectGlobal(enhancedStyles);
    }
}

// Create global instance
const emotion = new EmotionUtils();

// Initialize global styles and animations
emotion.createGlobalStyles();
emotion.createAnimations();
emotion.enhanceMarkdownStyles();

// Export for use in other modules
window.emotion = emotion; 