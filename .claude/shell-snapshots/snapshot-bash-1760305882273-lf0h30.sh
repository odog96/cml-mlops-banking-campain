# Snapshot file
# Unset all aliases to avoid conflicts with functions
unalias -a 2>/dev/null || true
shopt -s expand_aliases
# Check for rg availability
if ! command -v rg >/dev/null 2>&1; then
  alias rg='/home/cdsw/.nvm/versions/node/v22.20.0/lib/node_modules/\@anthropic-ai/claude-code/vendor/ripgrep/x64-linux/rg'
fi
export PATH=/home/cdsw/.nvm/versions/node/v22.20.0/bin\:/usr/lib/jvm/java-8-openjdk-amd64/bin\:/home/cdsw/.local/bin\:/usr/local/sbin\:/usr/local/bin\:/usr/sbin\:/usr/bin\:/sbin\:/bin\:/opt/conda/bin\:/usr/lib/hadoop/bin\:/usr/lib/hadoop-hdfs/bin\:/opt/spark/bin\:/opt/spark/bin
