<h1 align="center">
  GlobalQuake
</h1>

<!--<p align="center">
  <img src="" alt="GlobalQuake icon" title="GlobalQuake" />
</p>-->
<p align="center">
  <a href="https://github.com/xspanger3770/GlobalQuake/releases"><img src="https://img.shields.io/github/release/xspanger3770/GlobalQuake.svg?style=for-the-badge&logo=github" alt="Release"></a> <a href="https://github.com/xspanger3770/GlobalQuake/releases"><img src="https://img.shields.io/github/downloads/xspanger3770/GlobalQuake/total?style=for-the-badge&logo=github" alt="Releases"></a> <!--<a href="">--><img src="https://img.shields.io/badge/discord-Coming%20Soon...-blue?logo=discord&style=for-the-badge" alt="Discord"></a>
</p>

![GlobalQuake 0.9.0](https://github-production-user-asset-6210df.s3.amazonaws.com/100421968/259861791-6c41b8e4-d33c-44bc-a8ca-4f2ad7ecac40.png)

<div style="display: grid; grid-template-columns: 1fr 1fr;">
<img alt="StationManager" title="StationManager" src="https://github.com/xspanger3770/GlobalQuake/assets/100421968/a37319ec-2132-426a-b095-2e6a9e064322" style="width: 49%; height: auto;" />
<img alt="SourceManager" title="SourceManager" src="https://i.imgur.com/T1tmMtN.png" style="width: 49%; height: auto;" />
</div>

## Introduction

GlobalQuake is an experimental Java application that can be used to monitor earthquakes world-wide in near real time.

It enables selection of seismic stations downloaded from publicly available seismic networks via `fdsnws services` supllied by real time data from publicly available `seedlink servers`.\
The program uses this data to detect earthquakes and visualize them on an interactive 3D globe.\
It can also, estimate the earthquake magnitude using some empirical methods, but at the moment it can only work for small and medium size earthquakes, up to magnitude 5 or 6.

> [!WARNING]<br>
> Please keep in mind that GlobalQuake is still very experimental and should only be used for entertainment purposes, as the displayed information can be inaccurate or completly wrong.\
> Please be also aware that playing some of the included alarm sounds in public areas can be considered in some countries a form of fearmongering and illegal.\
> GlobalQuake doesn't own any form of data, and the respective owners can stop sharing them via Seedlink Server and/or FDSNWS at any moment without notice.

### What GlobalQuake is good at?

* It can easily select publicly available seismic stations.
* It can visualise detected earthquakes on a 3D global interactive map.
* It can quickly estimate small and moderate earthquake's magnitude, location and depth.

### What GlobalQuake is struggling with?

* Larger earthquakes (M6+) often trigger false detections or show duplicated earthquakes.
* Unable to detect multiple earthquakes in the same epicenter in short period of time.
* Calculation of distant earthquakes from a certain set of station is always less accurate than a local earthquake.

## System Requirements

- The system requirements will scale by the number of stations you select. This includes RAM, CPU and network usage.
- You can run GlobalQuake on slower system only to monitor earthquakes in your local area, or if your system can handle it, select hundreds or even thousands of stations around the world.
- Roughly speaking, 16GB of RAM, 6 CPU cores and 5Mbit network connection should be enough to handle about 1,000 stations.
- If GlobalQuake starts lagging heavily or even crashes after a few minutes, it is probably due to insufficient RAM in your system, and you need to select fewer stations.

## Download

* GlobalQuake comes as a standard executable JAR file. You can download the lastest version [here](https://github.com/xspanger3770/GlobalQuake/releases)
* Make sure you have the latest version of Java installed. [Download Java](https://www.oracle.com/java/technologies/downloads/)
* Put the JAR file to the location where you want the application data to be stored
* Run GlobalQuake by executing `java -jar [filename].jar` using your terminal/command line or simply double click it (Windows)
* You can also add `-Xms8G -Xmx8G` parameters and specify the maximum RAM allocation in gigabytes (replace the `8` with suitable value)
* Experimental: by adding `-Dsun.java2d.opengl=True`, you can improve your FPS

## Contributing

Any contributions, including feedback and suggestions are highly appreciated! See [Contributing guidelines](https://github.com/xspanger3770/GlobalQuake/blob/main/CONTRIBUTING.md)

## Project Licensing

This project is released under the terms of the MIT License.\
However, please note that this repository includes sound effects sourced from two other projects, each governed by their respective licenses.\
The sound effects with the [`LICENSE_J`](https://github.com/xspanger3770/GlobalQuake/blob/main/LICENSE_J) designation are used under the terms of their specific license - [JQuake](https://jquake.net/), and the sound effects with the [`LICENSE_K`](https://github.com/xspanger3770/GlobalQuake/blob/main/LICENSE_K) designation are also subject to their own unique license - [KiwiMonitor](https://kiwimonitor.amebaownd.com/).\
It's important to review and adhere to these additional licenses when using or distributing this project. Refer to the corresponding license files for more details.

## Special thanks to

![JQuake](https://images.weserv.nl/?url=avatars.githubusercontent.com/u/26931126?v=4&h=20&w=20&fit=cover&mask=circle&maxage=7d) [François Le Neindre](https://github.com/fleneindre) ([JQuake](https://jquake.net/en/)) - Inspiration for the layout, intensity scale, sound alarms and more\
![Philip Crotwell](https://images.weserv.nl/?url=avatars.githubusercontent.com/u/127367?v=4&h=20&w=20&fit=cover&mask=circle&maxage=7d) [Philip Crotwell](https://github.com/crotwell/) ([seisFile](http://crotwell.github.io/seisFile/), [TauP](http://crotwell.github.io/TauP/)) - Great and easy to use libraries. GlobalQuake wouldn't be possible without these\
![Yacine Boussoufa](https://images.weserv.nl/?url=avatars.githubusercontent.com/u/46266665?v=4&h=20&w=20&fit=cover&mask=circle&maxage=7d) [Yacine Boussoufa](https://github.com/YacineBoussoufa/) ([EarthquakeDataCenters](https://github.com/YacineBoussoufa/EarthquakeDataCenters)) - List of data providers for Seedlink and FDSNWS

### Contributors

<a href="https://github.com/xspanger3770/GlobalQuake/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xspanger3770/GlobalQuake" />
</a>
