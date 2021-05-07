def readFloatLines(filename, seperator = ' '):
    lines = []

    with open(filename) as fp:
        lines = fp.readlines()

    floatLines = []
    for line in lines:
        line = line.strip(' ')

        if len(line)<1:
            continue

        line_parts = line.split(seperator)

        floatLine = []
        for part in line_parts:
            floatLine.append(float(part))

        floatLines.append(floatLine)

    return floatLines

def readStrLines(filename, seperator = ' '):
    lines = []

    with open(filename) as fp:
        lines = fp.readlines()

    strLines = []
    for line in lines:
        line = line.strip(' ')

        if len(line) < 1:
            continue

        line_parts = line.split(seperator)

        strLine = []
        for part in line_parts:
            strLine.append(part)

        strLines.append(strLine)

    return strLines
