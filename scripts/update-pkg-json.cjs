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

// Write back
fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2));
console.log('Updated package.json files list');
