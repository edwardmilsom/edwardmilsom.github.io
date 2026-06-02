#!/bin/bash

# 1. Build the Homepage
echo "Building Homepage..."
pandoc index.md -o index.html \
  --template=template.html \
  --include-in-header="includes/head.html" \
  --include-before-body="includes/header.html" \
  --include-after-body="includes/footer.html" \
  --metadata pagetitle="Edward Milsom"

# 2. Build the Blog Posts (if any exist)
if find blog -mindepth 2 -maxdepth 2 -type f \( -name '*.md' -o -name '*.mdexample' \) 1> /dev/null 2>&1; then
    echo "Building Blog Posts..."
    
    # Start building the blog index content
    BLOG_INDEX_CONTENT="# All Blog Posts\n\n"
    
    while IFS= read -r file; do
        filename=$(basename -- "$file")
        dir=$(dirname -- "$file")
        name=$(basename -- "$dir")

        # Build the individual blog post into its own folder
        pandoc "$file" -o "blog/$name/index.html" \
          --template=template.html \
          --include-in-header="includes/head.html" \
          --include-before-body="includes/header.html" \
          --include-after-body="includes/footer.html" \
          --filter pandoc-katex
        
        echo "Converted $name"
        
        # Extract metadata for blog index
        TITLE=$(grep "^title:" "$file" | sed 's/title: *//')
        DATE=$(grep "^date:" "$file" | sed 's/date: *//')
        DESCRIPTION=$(grep "^description:" "$file" | sed 's/description: *//')
        
        # Add to blog index
        BLOG_INDEX_CONTENT+="* [**${TITLE}**]($name/index.html)  \n"
        BLOG_INDEX_CONTENT+="  <span class=\"date\">$DATE</span> - $DESCRIPTION\n\n"
    done < <(find blog -mindepth 2 -maxdepth 2 -type f \( -name '*.md' -o -name '*.mdexample' \) | sort)
    
    # Create blog index markdown file
    echo -e "$BLOG_INDEX_CONTENT" > blog/index.md
    
    # Build blog index HTML
    echo "Building Blog Index..."
    pandoc blog/index.md -o blog/index.html \
      --template=template.html \
      --include-in-header="includes/head.html" \
      --include-before-body="includes/header.html" \
      --include-after-body="includes/footer.html" \
      --metadata pagetitle="Blog - Edward Milsom"
fi

echo "Done."
