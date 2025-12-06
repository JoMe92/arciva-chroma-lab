/// <reference types="vitest" />
import { defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        include: ['quickfix-renderer/ts/**/*.test.ts'],
        environment: 'node',
        pool: 'forks',
    },
});
