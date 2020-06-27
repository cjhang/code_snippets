#!/usr/bin/env python

try:
    print(hello_string)
except:
    print('hello string is not defined')
    pass

if not hello_string:
    hello_string = 'hello world!'

print(hello_string)
