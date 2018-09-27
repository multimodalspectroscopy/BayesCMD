wget http://www.norg.uminho.pt/aivaz/pswarm/software/PPSwarm_v1_5.zip -O \
    tmp.zip && \
    unzip tmp.zip -d . && \
    rm tmp.zip && \
    # incorrect python reference inside Pswarm make file. Use SED to amend
    sed -i "s/python2.5/python2.7/g" "./PPSwarm_v1_5/makefile" && \
    sed -i "s/usr\/lib\/python2.7\/site-packages/\/home\/`whoami`\/.local\/lib\/python2.7\/site-packages/g" \
    "./PPSwarm_v1_5/makefile" && \
    cd ./PPSwarm_v1_5 && \
    make py && \

    cp pswarm_py.so ../batch/pylib/ && \
    cd .. && \
rm -r PPSwarm_v1_5
