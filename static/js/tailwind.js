tailwind.config = {
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                primary: '#8B5CF6',
                secondary: '#06B6D4',
                background: '#0A0F1C',
                surface: '#1A1F35',
                text: '#F8FAFC',
                accent: '#F59E0B',
                neutral: '#64748B',
                deep_gray: '#212f3d',
                main_content_bg: '#111928',
                gradient: {
                    start: '#8B5CF6',
                    middle: '#06B6D4',
                    end: '#F59E0B'
                }
            },
            fontFamily: {
                'inter': ['Inter', 'sans-serif'],
            },
            animation: {
                'float': 'float 6s ease-in-out infinite',
                'pulse-slow': 'pulse 3s ease-in-out infinite',
                'gradient': 'gradient 15s ease infinite',
                'glow': 'glow 2s ease-in-out infinite alternate',
                'slide-in': 'slideIn 0.5s ease-out',
                'bounce-subtle': 'bounceSubtle 2s ease-in-out infinite'
            },
            keyframes: {
                float: {
                    '0%, 100%': { transform: 'translateY(0px)' },
                    '50%': { transform: 'translateY(-20px)' }
                },
                gradient: {
                    '0%, 100%': { 'background-position': '0% 50%' },
                    '50%': { 'background-position': '100% 50%' }
                },
                glow: {
                    '0%': { 'box-shadow': '0 0 20px rgba(139, 92, 246, 0.5)' },
                    '100%': { 'box-shadow': '0 0 30px rgba(139, 92, 246, 0.8)' }
                },
                slideIn: {
                    '0%': { transform: 'translateX(-100%)', opacity: '0' },
                    '100%': { transform: 'translateX(0)', opacity: '1' }
                },
                bounceSubtle: {
                    '0%, 100%': { transform: 'translateY(0)' },
                    '50%': { transform: 'translateY(-5px)' }
                }
            },
            backgroundImage: {
                'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
                'gradient-mesh': 'linear-gradient(45deg, var(--tw-gradient-stops))',
                'aurora': 'linear-gradient(45deg, #8B5CF6, #06B6D4, #F59E0B, #8B5CF6)',
            }
        }
    }
}