ARCHIVE="http://www.cs-chan.com/source/ICIP2017/wikiart.zip"
FILENAME="wikiart.zip"
cd "data"
wget $ARCHIVE
unzip $FILENAME
rm -rf $FILENAME