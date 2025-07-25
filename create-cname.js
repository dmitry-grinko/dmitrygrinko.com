const fs = require('fs');
const path = require('path');

// Define the docs folder path and CNAME file path
const docsFolder = path.join(__dirname, 'docs');
const cnameFilePath = path.join(docsFolder, 'CNAME');

// Content for the CNAME file
const cnameContent = 'dmitrygrinko.com';

try {
  // Ensure the docs folder exists
  if (!fs.existsSync(docsFolder)) {
    fs.mkdirSync(docsFolder, { recursive: true });
    console.log('✅ Created docs folder');
  }

  // Write the CNAME file
  fs.writeFileSync(cnameFilePath, cnameContent);
  console.log('✅ Successfully created CNAME file in docs folder');
  console.log(`📁 File location: ${cnameFilePath}`);
  console.log(`📝 Content: ${cnameContent}`);
} catch (error) {
  console.error('❌ Error creating CNAME file:', error.message);
  process.exit(1);
} 