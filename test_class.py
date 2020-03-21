class test(object):
    def __init__(self, line):
        self.stuf_to = line 
    def print_line(self, stuff):
        print(stuff)
        print(self.stuf_to)
    def burrow(self, print_stuff):
        self.print_line("bye")
        print(print_stuff)