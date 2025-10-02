name: C99 commit build and Run

on:
  push:
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Build with gcc
      run: gcc -std=c99 -g -O0 -Wall commit.c -o commit -lssl -lcrypto
    - name: Run the commit program
      run: ./commit