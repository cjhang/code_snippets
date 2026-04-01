import math
import os
import time
import datetime
import re
import numpy 
import shutil
import sys

# Begin of main program
start = time.time()

proj = input("Project ID (e.g., w0a6 OR S14BE000):")

#check:
if (len(proj) != 4 and len(proj) != 8):
    print ("Wrong format of project ID, should be 4 characters")
    proj = input("Project ID (e.g., w0a6 OR S14BE000):")


if (len(proj) == 4 and proj[1] != "0"):
    if (proj[1] != "-"):
        print ("Please give main Project ID (e.g., w0a6), not subproject ID (waa6)")
        proj = input("Project ID (e.g., w0a6):")

if (len(proj) == 8 and proj[5:8] != "000"):
        print ("Please give main Project ID (e.g., w0a6 OR S14BE000), not subproject ID (waa6 OR S14B001)")
        proj = input("Project ID (e.g., w0a6 OR S14BE000):")
    
proj = proj.lower()


hdir = os.getenv("HOME")
#ipbdir = '/noemascratch/'+proj[:5]+'/DATA/'
#for multiprojects
dproj = hdir.split('/')[-1]
ipbdir = '/noemascratch/'+dproj+'/DATA/'

if not os.path.exists(ipbdir):
    #os.makedirs(ipbdir)
    gpfile = 'decide.txt'
    gpfil = open(gpfile,'w')
    out = 'y'
    outn = 'n'
    fmt =  '%'+str(len(out))+'s \n'            
    fmtn =  '%'+str(len(outn))+'s \n'            
    gpfil.writelines(fmt %(out))
    gpfil.writelines(fmt %(out))
    gpfil.writelines(fmtn %(outn))    
    gpfil.close()
    out = "getproj -p " + proj + " < decide.txt"
    os.system(out)
    os.remove(gpfile)


print ("Assuming IPB files are in: ",ipbdir)


if (len(proj) == 4):
    ipblist = [name for name in os.listdir(ipbdir) if (os.path.isfile(ipbdir+name) and name.split('.')[-1].upper() == "IPB" and name.split('.')[0][6:8] == proj[2:4].upper() and name.split('.')[0][4] == proj[0].upper())]
elif (len(proj) == 8):
    ipblist = [name for name in os.listdir(ipbdir) if (os.path.isfile(ipbdir+name) and name.split('.')[-1].upper() == "IPB" and name.split('.')[0][6:11] == proj[0:5].upper())]

 

print ("Found ",len(ipblist), " IPB files in ", ipbdir)


if not os.path.exists(hdir+'/calib'):
    print (hdir+'/calib', " does not exit, creating it...")
    os.makedirs(hdir+'/calib')

print ("Changing to: ", hdir+'/calib')
os.chdir(hdir+'/calib')


if (len(ipblist) == 0):
    print ("No IPB files found, running getproj...")
    gpfile = 'decide.txt'
    gpfil = open(gpfile,'w')
    out = 'y'
    fmt =  '%'+str(len(out))+'s \n'            
    gpfil.writelines(fmt %(out))
    gpfil.writelines(fmt %(out))
    gpfil.writelines(fmt %(out))    
    gpfil.close()
    out = "getproj -p " + proj + " < decide.txt"
    #out = "getproj < decide.txt"
    os.system(out)
    os.remove(gpfile)

    if (len(proj) == 4):
        ipblist = [name for name in os.listdir(ipbdir) if (os.path.isfile(ipbdir+name) and name.split('.')[-1].upper() == "IPB" and name.split('.')[0][6:8] == proj[2:4].upper() and name.split('.')[0][4] == proj[0].upper())]
    elif (len(proj) == 8):
        ipblist = [name for name in os.listdir(ipbdir) if (os.path.isfile(ipbdir+name) and name.split('.')[-1].upper() == "IPB" and name.split('.')[0][6:13] == proj[0:7].upper())]

    print ("Found ",len(ipblist), " IPB files in ", ipbdir)



oclicfi = "inp.clic"
oclicf = open(oclicfi,"w")

out = "sic out projdate.txt new"
fmt =  '%'+str(len(out))+'s \n'            
oclicf.writelines(fmt %(out))
out = "var gen"
fmt =  '%'+str(len(out))+'s \n'            
oclicf.writelines(fmt %(out)) 
for file in ipblist:
    out = "file in \"" + ipbdir+file + "\""
    fmt =  '%'+str(len(out))+'s \n'            
    oclicf.writelines(fmt %(out))
    out = "find /ty o /proc corr"
    fmt =  '%'+str(len(out))+'s \n'            
    oclicf.writelines(fmt %(out))
    out = "if (found.gt.28) then"
    fmt =  '%'+str(len(out))+'s \n'            
    oclicf.writelines(fmt %(out))
    out = "get f"
    fmt =  '%'+str(len(out))+'s \n'            
    oclicf.writelines(fmt %(out))
    out = "say 'project' 'date_observed' \"" +file + "\" /format a10 a14 a" +str(len(file)+2)
    fmt =  '%'+str(len(out))+'s \n'            
    oclicf.writelines(fmt %(out))
    out = "endif"
    fmt =  '%'+str(len(out))+'s \n'            
    oclicf.writelines(fmt %(out))

out = "sic out"
fmt =  '%'+str(len(out))+'s \n'            
oclicf.writelines(fmt %(out))

out = "exit"
fmt =  '%'+str(len(out))+'s \n'            
oclicf.writelines(fmt %(out))

oclicf.close()


out = 'gagiram; clic -nw @inp.clic'
os.system(out)

if os.path.isfile('inp.clic'): os.remove('inp.clic')


try:
    print ("Found ",sum(1 for line in open('projdate.txt') if line.rstrip()), " IPB files with corr scans on source")
except:
    print ("Something went wrong ... cannout count IPB files")


yes = set(['yes','y', 'ye', ''])
no = set(['no','n'])

choice = input('Do you want to use another IPB input list file ([yes]/no):').lower()
if choice in no:
    iclicfi = "projdate.txt"
    iclicf = open(iclicfi,"r")
    ipblist0 = [line.split() for line in iclicf]
    iclicf.close()
elif choice in yes:
    ipbnew = input('Please give full file name (same format as projdate.txt):')
    if os.path.isfile(ipbnew):
        ipbfile = open(ipbnew,"r")   
        ipblist0 = [line.split() for line in ipbfile]
        ipbfile.close()
    else:
        sys.exit("Your input file could not be found, exiting")
else:
   sys.exit("Please respond with 'yes' or 'no'")
   

ipblist = [l+[str(os.stat(ipbdir+l[2]).st_size/(1024.0*1024.0))] for l in ipblist0]

#print ipblist


l = [line[0] for line in ipblist]
count = [ (x,l.count(x)) for x in set(l)]

subp = ["0","-"]
for line in count:    
    if (len(line[0]) == 4):
        if (line[0][1].lower() not in subp):
            print ("Found following subprojects:", line[0]," #IPB files:",line[1])
            dir = hdir+'/calib/'+line[0].lower()
            if not os.path.exists(dir):
                os.makedirs(dir)
                print ("Creating directory: ",dir)
        else:
            dir = hdir+'/calib/'
    elif (len(line[0]) == 8):
        if (line[0][5:8] != "000"):
            print ("Found following subprojects:", line[0]," #IPB files:",line[1])
            dir = hdir+'/calib/'+line[0].lower()
            if not os.path.exists(dir):
                os.makedirs(dir)
                print ("Creating directory: ",dir)
        else:
            dir = hdir+'/calib/'
            

if (len(ipblist) > 0): 

    if os.path.isfile('locpipe0.clic'): os.remove('locpipe0.clic')
    if os.path.isfile('locpipe.clic'): os.remove('locpipe.clic')
    orgpip = "/users/softs/gildas/gildas/gildas-exe-last/pro/pipeline.clic"
    shutil.copy2(orgpip, 'locpipe0.clic')
    os.chmod('./locpipe0.clic',0o666)

    oclicfi = "locpipe0.clic"
    oclicf = open(oclicfi,"a")
    oclicf.write("exit \n")
    oclicf.close()

    #oclicfi = "locpipe.clic"
    #oclicf = open(oclicfi,"w")
    #oclicf.write("on error exit \n")
    #oclicf.close()

    out = "cat locpipe0.clic >> locpipe.clic"
    os.system(out)

    if os.path.isfile("finished-pipelines.dat"): os.remove("finished-pipelines.dat")


    cc = 0
    tf = 0
    nf = 0

    for line in ipblist:
        if (len(line[0]) == 4 and line[0][1].lower() not in subp):
            dir = hdir+'/calib/'+line[0].lower()
            print ("Changing to: ", dir)
            os.chdir(dir)
            shutil.copy2(hdir+'/calib/locpipe.clic', 'locpipe.clic')
        if (len(line[0]) == 8 and line[0][5:8] != "000"):
            dir = hdir+'/calib/'+line[0].lower()
            print ("Changing to: ", dir)
            os.chdir(dir)
            shutil.copy2(hdir+'/calib/locpipe.clic', 'locpipe.clic')



        print ("Running pipeline for: ",line[0],line[1])
        hpbf = line[1].lower()+"-"+line[0].lower()+".hpb"
        hpbc = line[1].lower()+"-"+line[0].lower()+"clic"
        dorepeat = 0
        if os.path.isfile(hpbf):
            rmhpb = input("An hpb file on this track already exists, do you want to remove it ([yes]/no)?")            
            if (rmhpb.lower() in yes):
                print ("Removing ", hpbf, " and ", hpbc)
                os.remove(hpbf)
                if os.path.isfile(hpbc): os.remove(hpbc)
            elif (rmhpb in no):
                alter = input("Do you want to rerun pipeline on existing hpb file ([yes]/no)?")
                if (alter.lower() in yes):
                    dorepeat = 1
                else:
                    print ("Sorry do not understand what you want, exiting...")
                    sys.exit()
            else:
                print ("Reply must be yes or no, you said ", rmhpb,", exiting...")
                sys.exit()
                


                        
        out = "gagiram; clic < locpipe1.clic " 
        repfi = "repeat.txt"
        if (dorepeat == 1):
            repf = open(repfi,"w")
            out2 = "repeat"
            fmt =  '%'+str(len(out2))+'s \n'            
            repf.write(fmt %(out2))
            repf.close()            
            # This does not work anymore!
            # out = out + " < repeat.txt "
            print (out)
        out = out + " > locpipe-" + line[0] +"-" + line[1] + ".log"
        print (out)
        
        l3 = float(line[3])
        ofile = open('locpipe1.clic','w')
        if (l3*2/1024.0 <= 360. and l3*2/1024.0 >= 80.):
            ofile.writelines('sic log space_clic ' + '%i5'%(int(l3*2.0/1024.0+30.0)) + ' \n')
        ofile.writelines('@ locpipe ' + line[0] +" " + line[1] + '\n')
        ofile.close()

        dir = hdir+'/calib/'

        if (l3*2/1024.0 > 360.0):
            print ("Cannot run pipeline as filesize exceeds 180GB: ", '%i5'%(l3/1024.0))
            os.chdir(dir)
            oclicfi = "filesizetoobig.dat"
            oclicf = open(oclicfi,"a")
            out = " ".join(line)
            fmt =  '%'+str(len(out))+'s \n'            
            oclicf.write(fmt %(out))
            oclicf.close()
            tf = tf + 1
        else:
            try:
                if os.system(out) != 0:
                    raise Exception('clic command does not exist')
                print("Success running clic")
                if os.path.isfile(repfi): os.remove(repfi)
                os.chdir(dir)
                oclicfi = "finished-pipelines.dat"
                oclicf = open(oclicfi,"a")
                out = " ".join(line)
                fmt =  '%'+str(len(out))+'s \n'            
                oclicf.write(fmt %(out))
                oclicf.close()
                cc = cc + 1
                print ("Finished ", cc, " of ", len(ipblist), "IPB files")
            except:
                print("Some error running pipeline")
                xout = "tail -20 "+ line[0] +"-" + line[1] + ".log"
                os.system(xout)
                os.chdir(dir)
                oclicfi = "notfinished-pipelines.dat"
                oclicf = open(oclicfi,"a")
                out = " ".join(line)
                fmt =  '%'+str(len(out))+'s \n'            
                oclicf.write(fmt %(out))
                oclicf.close()
                nf = nf + 1
                print ("Number of crashed pipelines: ", nf)



        
else:
    print ("No IPB files found with data in it, please check ..")


print()
print ("Finished ", cc, " of ", len(ipblist), "IPB files; crashed: ", nf, "; filesize exceeding 160GB for ", tf, " files")
print()


stop = time.time()
dure = stop - start
print ("Run time = ",dure, "seconds")
