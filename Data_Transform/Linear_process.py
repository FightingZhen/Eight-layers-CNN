filename='Ret_GGO_'
filenumber=35

def process_accelerate_k(filenumber):
    global filename
    max=0.0
    min=0.0

    if filenumber<9:
        file_read=open('D:\\SourceData\\' + filename + '00' + str(filenumber+1) + '.txt')
    else:
        file_read  = open('D:\\SourceData\\' + filename + '0' + str(filenumber+1) + '.txt')

    for line in file_read:
        if line.index('\n')!=1:
            line=line[:-2]
            line=line.split(',')
            for item in line:
                if float(item)<min:
                    min=float(item)
                elif float(item)>max:
                    max=float(item)
                else: pass
    k=1.0/(max-min)
    b=1.0-max*k

    file_read.close()
    return k,b

def process_result(filenumber):
    global filename

    k,b=process_accelerate_k(filenumber)
    w_array=[]

    if filenumber < 9:
        file_read  = open('D:\\SourceData\\' + filename + '00' + str(filenumber+1) + '.txt')
        file_write = open('D:\\Linear_Result\\'+ filename + '00' + str(filenumber+1) + '.txt', 'w')
    else:
        file_read  = open('D:\\SourceData\\' + filename + '0' + str(filenumber+1) + '.txt')
        file_write = open('D:\\Linear_Result\\' + filename + '0' + str(filenumber + 1) + '.txt', 'w')

    for line_read in file_read:
        if line_read.index('\n')==1:
            file_write.write(line_read)
        else:
            line_read=line_read[:-2]
            line_read=line_read.split(',')
            for item in line_read:
                result=k*float(item)+b
                w_array.append(str(result))
            write_str=','.join(w_array)
            file_write.write(write_str+'\n')
            w_array=[]
    file_read.close()
    file_write.close()

def main():
    for i in range(filenumber):
        print("Picture %d begin !" % i)
        process_result(i)
        print("Picture %d finish !" % i)

if __name__ == '__main__':
    main()