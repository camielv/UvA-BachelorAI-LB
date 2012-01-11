# Extractor extracts a column from a csv file
import csv

file = csv.reader(open('DataCSV.csv', 'rb'), delimiter=',', quotechar='"')


attributes = []
i = 0

for row in file:
    
    attributes.append( row[9] )

    # Stop at 20
    i += 1
    if ( i > 20 ): break

print '\n'.join(attributes)
