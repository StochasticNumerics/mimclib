#!/bin/bash

# This script is used to install mimclib and its main dependencies
# on a Debian based machine.

# The prerequisite of this is a Debian system and root access using
# sudo.

# This script  will upgrade your existing setup prior to adding extra
# packages. If you are not happy with this, you might want to modify this
# script. Comments and versions of this script for other platforms
# are welcome in the form of github issues, pull requests or
# emails to juho.happola@iki.fi

# The script might not run without supervision. Unless you have mysql server
# installed already, it will prompt for a mysql root password.
# Also, depending on fast the required files are being downloaded
# you might need to enter root password a few times. The runtime
# of this script might be around an hour, depending on your
# network speed and which packages need to be upgraded and installed.

# This has been tried out with Debian Jessy with the May 22th 2016 (AMD64)
# and the following commit of the library:
# d94151770d87f4f685efc3719ba3b08249d4e5ae

echo -e "First, update existing packages..."

sudo apt-get update
sudo apt-get upgrade

echo -e "Existing packages upgraded."
echo -e "Installing packages..."

sudo apt-get install git python-pip mysql-server mysql-client libmysqlclient-dev build-essential ipython libpng-dev libfreetype6-dev libxft-dev libpython-dev liblapack-dev libblas-dev gfortran parallel

echo -e "Done installing aptitude packages."

echo -e "Adding python packages using pip"

pip install --upgrade pip

# For whatever reason this needs root privileges
sudo -H pip install --upgrade numpy
sudo -H pip install --upgrade matplotlib
sudo -H pip install --upgrade scipy
sudo -H pip install --upgrade mysql-python

echo -e "Done adding python packages"

cd
git clone https://github.com/StochasticNumerics/mimclib.git
cd mimclib
make
make pip

echo -e "Creating the mimc database to mySQL.\n You will need to enter mySQL root password."
python -c 'from mimclib.db import MIMCDatabase; print MIMCDatabase().DBCreationScript();' | mysql -u root -p

echo -e "Adding a user to mySQL:"
echo -e "CREATE USER '$USER'@'%';\nGRANT ALL PRIVILEGES ON mimc.* TO '$USER'@'%' WITH GRANT OPTION;" | mysql -u root -p

cd tests/gbm
make


