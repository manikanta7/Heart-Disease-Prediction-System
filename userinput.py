age = raw_input("What is your age? ")
total_cholstral = raw_input("What is your cholastral? ")
heartrate = raw_input("What is your heartrate? ")
restedbps = raw_input("What is your restbps? ")
yersofsmoke = raw_input("What is your smoke? ")
cigerateperday = raw_input("What is your perday? ")
exercise_st = raw_input("What is your excersice? ")
exbp = raw_input("What is your exbp? ")
maxhrt = raw_input("What is your maxhrt? ")
mets = raw_input("What is your mets? ")
count=0;
if age>30:
    count=count+1;
if total_cholstral>250:
    count=count+1
if heartrate>85:
    count=count+1
if restedbps>120:
    count=count+1
if yersofsmoke>2:
    count=count+1
if cigerateperday>3:
    count=count+1
if exercise_st>3:
    count=count+1
if exbp>125:
    count=count+1
if maxhrt>90:
    count=count+1
if maxhrt<60:
    count=count+1
if count>5:
    print 'Chance To Heart Attack'
if count<5:
    print 'No Chance To Heart Attack'
    






















