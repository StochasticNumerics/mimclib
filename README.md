User notes from Alexander Litvinenko:

First time command make didn't work The reason was an old gcc compiler. 

I typed

`module unload gcc`

`module load gcc/5.1.0`

`make`

It worked.

I tried to install mysql and failed

`python -c "from mimclib.db import MIMCDatabase ; print MIMCDatabase().DBCreationScript();" | mysql`

ERROR 1045 (28000): Access denied for user 'litvina'@'localhost' (using password: NO)

close failed in file object destructor:

sys.excepthook is missing

lost sys.stderr

I installed first

`sudo apt-get install mysql-server`

Tried again and failed again

`python -c "from mimclib.db import MIMCDatabase ; print MIMCDatabase().DBCreationScript();" | mysql`

ERROR 1045 (28000): Access denied for user 'litvina'@'localhost' (using password: NO)

close failed in file object destructor:

sys.excepthook is missing

lost sys.stderr

Correct solution is

`mysql -u root -p`

Enter password:

Welcome to the MySQL monitor.  Commands end with ; or \g.


`grant all privileges on *.* to 'litvina'@'%' with grant option;`

Query OK, 0 rows affected (0.00 sec)



Then

`module unload gcc`

Unloading module gcc version 4.6.0

Initial gcc version: 4.6.0

Current gcc version: 4.8.4-2ubuntu1~14.04.1)

`module load gcc/5.1.0`

Loading module gcc version 5.1.0

Initial gcc version: 4.8.4-2ubuntu1~14.04.1)

/opt/share/gcc/5.1.0

Current gcc version: 5.1.0

`python -c "from mimclib.db import MIMCDatabase ; print MIMCDatabase().DBCreationScript();" | mysql`

It seems that installation of mimclib is successful.

