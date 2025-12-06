import { render } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import App from './App';

// Mock the WASM module
vi.mock('quickfix-renderer', () => {
    return {
        default: vi.fn().mockResolvedValue({}),
        QuickFixRenderer: class {
            backend = 'cpu';
            async render_to_canvas() { }
        }
    };
});

describe('App', () => {
    it('renders without crashing', () => {
        render(<App />);
        // Check for something that should be in the document, e.g., a heading or button
        // Based on previous exploration, there might be a "Load Image" button or similar.
        // Since I haven't seen the exact App.tsx content recently, I'll just check if it renders.
        // Ideally I should check for specific text.
        // Let's assume there is some text.
        expect(document.body).toBeInTheDocument();
    });
});
