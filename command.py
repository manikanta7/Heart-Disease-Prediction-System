import csv
with open('data.csv', 'r') as f:
	data_dt = list(csv.reader(f, delimiter=';'))
	print(data_dt[:3])
with open("data.csv", 'r') as f:
	qualities = [float(item[1])
				 #for item in data_dt[1:]]:
	#sum(qualities) / len(qualities)
				 
age = input("What is your age? ")
total_cholstral = raw_input("What is your cholastral? ")
heartrate = raw_input("What is your heartrate? ")
restedbps = raw_input("What is your restbps? ")
yersofsmoke = raw_input("What is your smoke? ")
cigerateperday = raw_input("What is your perday? ")
exercise_st = raw_input("What is your excersice? ")
exbp = raw_input("What is your exbp? ")
maxhrt = raw_input("What is your maxhrt? ")
mets = raw_input("What is your mets? ")
print((int(age)+int(total_cholstral)+int(heartrate)+int(restedbps)+int(yearsofsmoke)+int(cigerateperday)+int(exercise_st)+int(exbp)+int(maxhrt)+int(mets))/n)
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
