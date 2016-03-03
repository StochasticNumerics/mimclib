import mimclib.plot as miplot
import mimclib.db as mimcdb
import matplotlib.pyplot as plt

db = mimcdb.MIMCDatabase()
run_data = db.readRunData(db.getRunDataIDs(tag="NoTag"))

fig = plt.figure()
# plt.figure(figsize=(10,6))
miplot.plotTOLvsErrors(fig.gca(), run_data, exact=0.0)
plt.savefig('TOLvsErrors.pdf',dpi=500)
# plt.show()

fig = plt.figure()
# plt.figure(figsize=(10,6))
miplot.plotLvlsVsTime(fig.gca(), run_data, '-o')
plt.savefig('TimeVsLvls.pdf',dpi=500)
# plt.show()

fig = plt.figure()
# plt.figure(figsize=(10,6))
miplot.plotExpectVsLvls(fig.gca(), run_data, fmt='-o')
plt.savefig('ExpectVsLvls.pdf',dpi=500)
# plt.show()

fig = plt.figure()
# plt.figure(figsize=(10,6))
miplot.plotVarVsLvls(fig.gca(), run_data, '-o')
plt.savefig('VarVsLvls.pdf',dpi=500)
# plt.show()

fig = plt.figure()
# plt.figure(figsize=(10,6))
miplot.plotTimeVsTOL(fig.gca(), run_data, fmt='-o')
plt.savefig('TimeVsTOL.pdf',dpi=500)
# plt.show()

fig = plt.figure()
# plt.figure(figsize=(10,6))
miplot.plotLvlsVsTOL(fig.gca(), run_data, '-o')
plt.savefig('LvlsVsTOL.pdf',dpi=500)
# plt.show()

fig = plt.figure()
# plt.figure(figsize=(10,6))
miplot.plotErrorsQQ(fig.gca(), run_data, '-o', tol=0.001)
plt.savefig('QQ.pdf',dpi=500)
# # plt.show()

plt.show()
