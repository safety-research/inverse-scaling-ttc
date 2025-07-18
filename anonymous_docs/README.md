# Anonymous Hosting Instructions

This folder contains an anonymized version of the inverse scaling demo for paper review.

**Anonymous Repository URL:** https://anonymous.4open.science/r/inverse_scaling-C88D

## Files Included

- `index.html` - Main demo page (anonymized)
- `demo_data.js` - Complete dataset with all examples
- `script.js` - Interactive functionality
- `README.md` - This file

## Hosting Options

### Option 1: GitHub Pages (Recommended)
1. Create a new GitHub account with anonymous email
2. Create a new repository (e.g., `inverse-scaling-demo`)
3. Upload these files to the repository
4. Enable GitHub Pages in Settings > Pages
5. Your demo will be available at: `https://[username].github.io/inverse-scaling-demo/`

### Option 2: Netlify Drop
1. Go to https://app.netlify.com/drop
2. Drag and drop this entire folder
3. Get instant anonymous URL
4. No account needed

### Option 3: Surge.sh
```bash
npm install -g surge
cd anonymous_docs
surge
# Choose a random subdomain when prompted
```

### Option 4: University Server
- Upload to your institution's anonymous hosting service
- Ensure no identifying information in the URL path

## Local Testing
```bash
# Python 3
python -m http.server 8000

# Or use the included serve script
python serve.py
```

Then open http://localhost:8000 in your browser.

## Important Notes

- All author information has been removed
- No analytics or tracking scripts included
- No external dependencies except Google Fonts
- Fully self-contained demo

## For Reviewers

This interactive demo shows 24 instances of inverse scaling across:
- 8 different task types
- 4 language models
- Multiple reasoning budgets (0 to 16384 tokens)

Use the dropdown menus to explore different examples and see how model performance degrades with increased reasoning.