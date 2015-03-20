# coding=utf-8
from collections import defaultdict
import os
import sys
import errno
import pypyodbc

if __name__ == "__main__":
    genders = dict()
    filename = sys.argv[1]
    with open(filename) as f:
        data = f.readlines()
        for line in data:
            spl = line.split()
            genders[spl[0]] = spl[1:]
    conn = pypyodbc.connect(
    r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
    r"Dbq=C:\Users\rhdgjest\Documents\censusscan\data\RestVestfold"
    r".accdb;")
    cursor = conn.cursor()
    query = u"""
        SELECT IMAGE_ID, PR_PERSON_NBR, PR_SEX_CODE FROM main
        WHERE EVENT_CLERICAL_DISTRICT='{}' AND (MULTI_RECORD_TYPE)='TYPE 1'
        ORDER BY IMAGE_ID;"""
    district = os.path.splitext(filename)[0].split("_")[1]
    cursor.execute(query.format(district))
    persons = defaultdict(list)
    absentfile = os.path.join(
        r"C:\Users\rhdgjest\Documents\absent", district + ".txt")
    try:
        with open(absentfile) as f:
            absent = [x.split()[0] for x in f]
    except IOError as e:
        absent = None
        if e.errno != errno.ENOENT:
            raise
    for person in cursor:
        image_name = os.path.split(person[0])[1]
        persons[image_name].append((person[1], person[2]))
    for v in persons.itervalues():
        v.sort(key=lambda x: int(x[0]))
    output = []
    for image, v in persons.iteritems():
        if image in absent:
            output.append("{} {}\n".format(image, "f"))
            continue
        if len(genders[image]) != len(v):
            print image + " mismatch"
    print "teh"