### User notes from Alexander Litvinenko:

First time command make didn't work The reason was an old gcc compiler. 

I typed

`module unload gcc`
`module load gcc/5.1.0`
`make`

It worked.

I tried to install mysql and failed

`python -c "from mimclib.db import MIMCDatabase ; print MIMCDatabase().DBCreationScript();" | mysql`

>ERROR 1045 (28000): Access denied for user 'litvina'@'localhost' (using password: NO) <br />
>close failed in file object destructor: <br />
>sys.excepthook is missing <br />
>lost sys.stderr 

I installed first

`sudo apt-get install mysql-server`

Tried again and failed again

`python -c "from mimclib.db import MIMCDatabase ; print MIMCDatabase().DBCreationScript();" | mysql`

>ERROR 1045 (28000): Access denied for user 'litvina'@'localhost' (using password: NO) <br />
>close failed in file object destructor: <br />
>sys.excepthook is missing <br />
>lost sys.stderr 

Correct solution is

`mysql -u root -p`

>Enter password: <br />
>Welcome to the MySQL monitor.  Commands end with ; or \g.


`grant all privileges on *.* to 'litvina'@'%' with grant option;`

>Query OK, 0 rows affected (0.00 sec)



Then

`module unload gcc`

>Unloading module gcc version 4.6.0
>Initial gcc version: 4.6.0
>Current gcc version: 4.8.4-2ubuntu1~14.04.1)

`module load gcc/5.1.0`

>Loading module gcc version 5.1.0 <br />
>Initial gcc version: 4.8.4-2ubuntu1~14.04.1) <br />
>/opt/share/gcc/5.1.0 <br />
>Current gcc version: 5.1.0 <br />

`python -c "from mimclib.db import MIMCDatabase ; print MIMCDatabase().DBCreationScript();" | mysql`

It seems that installation of mimclib is successful.

To be able to run tests/pde example, you have to:
1) Create by hand libraries. Go to each directory which contain file makefile and run command make. These libraries will be necessary to compile  tests/pde example.
2) Change everywhere "abdo" onto your MYSQL login name. Start with file tests/pde/run.py  . You may need to recompile everything after changes.

