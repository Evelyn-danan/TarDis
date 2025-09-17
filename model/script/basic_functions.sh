#!/bin/bash
##basic functions
create_dir(){
    if [ -z ${1} ];then
       echo Please enter a path to create 
       exit -1
    fi
    for i in $@
    do
        if [ ! -d ${i} ];then
           mkdir -p ${i}
        fi
    done
}

auto_createdir(){
    if [ -z ${1} ];then
        echo Please enter a path to create 
        exit -1
    fi
    for i in $@
    do
        if [ -d ${i} ];then
            rm -rf ${i}
        fi
        mkdir ${i}
    done
}

del_dir(){
    if [ -z ${1} ];then
    echo Please enter a path to del 
    exit -1
    fi
    for i in $@
    do
        if [ -d ${i} ];then
            rm -rf ${i}
        fi
    done
}

auto_touchfile(){
    if [ -z ${1} ];then
        echo Please enter a file to touch
        exit -1
    fi
    for i in $@
    do
        if test -e ${i}
        then
            rm -rf ${i}
        fi
        touch ${i}
    done
}

del_file(){
    if [ -z ${1} ];then
        echo Please enter a file to del
        exit -1
    fi
    for i in $@
    do
        if test -e ${i}
        then
            rm -rf ${i}
        fi
    done
}
