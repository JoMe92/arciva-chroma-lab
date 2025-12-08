const fs = require('fs');
const path = require('path');

const pkgPath = path.join(__dirname, '../quickfix-renderer/pkg/package.json');
const pkg = require(pkgPath);

// Add new files to 'files' list
const newFiles = [
    'client.js',
    'client.d.ts',
    'worker.js',
    'worker.d.ts',
    'protocol.js',
    'protocol.d.ts'
];

if (pkg.files) {
    newFiles.forEach(f => {
        if (!pkg.files.includes(f)) {
            pkg.files.push(f);
        }
    });
}

// Add publishConfig for GitHub Packages
pkg.publishConfig = {
    "registry": "https://npm.pkg.github.com"
};

// Add main and type for better compatibility
pkg.main = "quickfix_renderer.js";
pkg.type = "module";

// Write back
fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2));
console.log('Updated package.json files list');
