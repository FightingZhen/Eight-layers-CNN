import numpy as np
filename='Mul_CON_'
filenumber=35

def process_average(filenumber):
    global filename
    average = 0
    number_pixel = 0

    if filenumber < 9:
        file_read  = open('D:\\SourceData\\' + filename + '00' + str(filenumber+1) + '.txt')
    else:
        file_read  = open('D:\\SourceData\\' + filename + '0' + str(filenumber+1) + '.txt')

    for line_avg in file_read:
        if line_avg.index('\n')!=1:
            line_avg=line_avg[:-2]
            line_avg_split=line_avg.split(',')
            for item in line_avg_split:
                average+=int(item)
                number_pixel+=1
    average/=number_pixel
    file_read.close()
    return average

def process_variance(filenumber):
    global filename
    variance=0
    number_pixel=0
    average=process_average(filenumber)

    if filenumber < 9:
        file_read  = open('D:\\SourceData\\' + filename + '00' + str(filenumber+1) + '.txt')
    else:
        file_read  = open('D:\\SourceData\\' + filename + '0' + str(filenumber+1) + '.txt')

    for line_variance in file_read:
        if line_variance.index('\n')!=1:
            line_variance = line_variance[:-2]
            line_var_split=line_variance.split(',')
            for item in line_var_split:
                variance+=(int(item)-average)*(int(item)-average)
                number_pixel += 1
    variance/=number_pixel
    file_read.close()
    return variance

def process_result(filenumber):
    global filename
    average=process_average(filenumber)
    variance=process_variance(filenumber)
    w_array=[]

    if filenumber < 9:
        file_read  = open('D:\\SourceData\\' + filename + '00' + str(filenumber+1) + '.txt')
        file_write = open('D:\\Avg_Var_Result\\'+ filename + '00' + str(filenumber+1) + '.txt', 'w')
    else:
        file_read  = open('D:\\SourceData\\' + filename + '0' + str(filenumber+1) + '.txt')
        file_write = open('D:\\Avg_Var_Result\\' + filename + '0' + str(filenumber + 1) + '.txt', 'w')

    for line in file_read:
        if line.index('\n')==1:
            file_write.write(line)
        else:
            line=line[:-2]
            line_split=line.split(',')
            for item in line_split:
                n_item=(int(item)-average)/np.sqrt(variance)
                w_array.append(str(n_item))
            str_array=','.join(w_array)
            file_write.write(str_array+'\n')
            w_array=[]
    file_read.close()
    file_write.close()

def main():
    global filenumber
    for i in range(filenumber):
        print("Picture %d begin !" % i)
        process_result(i)
        print("Picture %d finished !" % i)

if __name__ == '__main__':
    main()