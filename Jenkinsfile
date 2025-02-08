{\rtf1\ansi\ansicpg1252\cocoartf2709
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\froman\fcharset0 Times-Italic;\f1\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red236\green236\blue241;\red236\green236\blue241;\red100\green100\blue100;
\red98\green9\blue1;\red251\green0\blue7;\red210\green9\blue5;}
{\*\expandedcolortbl;;\cssrgb\c94118\c94118\c95686;\cssrgb\c94118\c94118\c95686\c392;\cssrgb\c46667\c46667\c46667;
\cssrgb\c46667\c6667\c0;\cssrgb\c100000\c0\c0\c5098;\cssrgb\c86667\c13333\c0;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\i\fs32 \cf2 \cb3 \expnd0\expndtw0\kerning0
Jenkinsfile (Declarative Pipeline)\cb1 \
\pard\pardeftab720\partightenfactor0

\f1\i0\fs28 \cf4 \cb3 /* Requires the Docker Pipeline plugin */\cf2 \
pipeline \{\
    agent \{ docker \{ image \cf5 \cb6 '\cf7 python:3.13.2-alpine3.21\cf5 '\cf2 \cb3  \} \}\
    stages \{\
        stage(\cf5 \cb6 '\cf7 build\cf5 '\cf2 \cb3 ) \{\
            steps \{\
                sh \cf5 \cb6 '\cf7 python --version\cf5 '\cf2 \cb3 \
            \}\
        \}\
    \}\
\}}