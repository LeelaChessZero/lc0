---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**WARNING: Please only use GitHub issues for BUG REPORTS and FEATURE REQUESTS**
Lengthy algorithm discussions etc are fine too, but also consider creating a wiki page or a page on website.

Here is what to do with other types of questions:
1. Build problems: ask in #help channel in our discord: https://discord.gg/pKujYxD
2. Configuration questions: ask in #help channel in our discord: https://discord.gg/pKujYxD
3. Running problems: ask in #help channel in our discord: https://discord.gg/pKujYxD


If you are filing a bug report, please fill the fields below.
Otherwise, feel free to remove this text and type a free-form issue as usual.

BUG REPORT

**Describe the bug**
A clear and concise description of what the bug is.

**Steps to Reproduce**
1. 
2. 
3. 
4. 
Expected behavior:
Observed behavior:

**Lc0 version**
Include Lc0 version/operating system/backend type.

**Lc0 parameters**
Command line, if not default.
Include screenshot of configuration window, if using through GUI.

**Hardware**
* Number and model of GPUs that you use.
* Amount of RAM in the system
* Other specs (CPU etc) if it may be relevant

**Lc0 logs**
Please attach Lc0 logs. Here is how to produce them (e.g. for D:\logfile.txt):

Set the following UCI option:
**Logfile:** D:\\logfile.txt
OR
pass this as a command line argument:
`--logfile=D:\logfile.txt`
OR
Create **lc0.config** file in the same directory as your **lc0.exe** is located, with the following contents:
```
logfile=D:\logfile.txt
```

After running Lc0, **D:\logfile.txt** should appear.


**Chess GUI logs**
If there is a problem with particular GUI (cutechess/arena/etc), also attach logs of that program.
