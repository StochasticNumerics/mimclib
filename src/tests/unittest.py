import unittest

class TestStringMethods(unittest.TestCase):
    def test_pde(self):
        import pde.run
        # TODO:  Figure out the correct value and put it here
        # TODO: Figure out a way to pass the correct arguments
        self.assertEqual(pde.run.MLMCPDE(), 0)

if __name__ == '__main__':
    unittest.main()
