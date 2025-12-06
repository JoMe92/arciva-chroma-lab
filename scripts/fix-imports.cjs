const fs = require('fs');
const path = require('path');

const pkgDir = path.join(__dirname, '../quickfix-renderer/pkg');
const filesToFix = ['worker.js', 'client.js'];

filesToFix.forEach(file => {
    const filePath = path.join(pkgDir, file);
    if (fs.existsSync(filePath)) {
        let content = fs.readFileSync(filePath, 'utf8');

        // Replace incorrect relative path from tsc output with sibling path suitable for flat package
        // Old: import ... from '../pkg/quickfix_renderer';
        // New: import ... from './quickfix_renderer.js';
        const newContent = content.replace(/from\s+['"]\.\.\/pkg\/quickfix_renderer['"]/g, "from './quickfix_renderer.js'");

        if (content !== newContent) {
            fs.writeFileSync(filePath, newContent, 'utf8');
            console.log(`Fixed imports in ${file}`);
        } else {
            console.log(`No imports needed fixing in ${file}`);
        }
    } else {
        console.warn(`File not found: ${filePath}`);
    }
});
