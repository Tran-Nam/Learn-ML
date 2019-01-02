import urllib
source = urllib.urlretrieve('http://anhsv.hust.edu.vn/Student/20152610.jpg')
output = open('test.png', 'wb')
output.write(source.read())
output.close()