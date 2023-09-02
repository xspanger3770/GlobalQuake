# Downloading and Installation

## Table of Contents

| [Java Download](#JavaDownload)  | [Java Install](#JavaInstall)   | [Setting Environment Variables](#Environment) | [GlobalQuake](#GlobalQuake)      |
|---------------------------------|--------------------------------|-----------------------------------------------|----------------------------------|
| [Windows](#JavaDownloadWindows) | [Windows](#JavaInstallWindows) | [Windows](#EnvironmentWindows)                | [Download](#GlobalQuakeDownload) |
| [Linux](#JavaDownloadLinux)     | [Linux](#JavaInstallLinux)     | [Linux](#EnvironmentLinux)                    | [Install](#GlobalQuakeInstall)   |
| [MacOS](#JavaDownloadMacOS)     | [MacOS](#JavaInstallMacOS)     | [MacOS](#EnvironmentMacOS)                    | [Running](#GlobalQuakeRun)       |                


## Java
<a name="Java"></a>
(If you have Java already installed skip to the [GlobalQuake](#GlobalQuake) Section)

> [!NOTE]<br>
> There are 2 architectures, x86-64 (also referred to as just "x64") and arm64, generally most computers have x86-64 so this tutorial will assume your device has that.
> If you encounter any errors make sure you have the correct Java for your architecture.\
> If you're on a recent MacOS device you should make sure of your architecture since some MacOS devices use arm64.**

### Downloading
<a name="JavaDownload"><a/>
Download Java 17.
You can download Java for:

- [Windows](https://www.oracle.com/java/technologies/downloads/#jdk17-windows)
- [Linux](https://www.oracle.com/java/technologies/downloads/#jdk17-linux)
- [MacOS](https://www.oracle.com/java/technologies/downloads/#jdk17-mac)

#### Windows
<a name="JavaDownloadWindows"></a>
For Windows either the x64 Installer or x64 MSI Installer is fine to use.

#### Linux
<a name="JavaDownloadLinux"></a>
For Linux it highly depends on the distribution you are running. If you are running Debian, or a distribution based on Debian like Ubuntu, download the **_x64 Debian Package_**. If your distro is based off of Red Hat like Rocky Linux, AlmaLinux etc, use the "**_x64 RPM Package_**".

#### MacOS
<a name="JavaDownloadMacOS"></a>
For MacOS download the "**_x64 DMG Installer_**".

### Installing
<a name="JavaInstall"></a>
#### Windows
<a name="JavaInstallWindows"></a>
1. Download the installer
2. Open the installer
3. Click the "**_Next_**" button on the first Screen
4. Choosing a custom install location (if you don't want a custom install location skip to step )
5. Go to the location you want to install Java with the file explorer. You can choose another drive by clicking the topdown menu at the top with the little downwards arrow on it. You can add folders using the button on the top right with the folder and shine icon. For example I went to my D drive and created a folder called JDKs and a sub folder called 17, for the JDK 17 version, for the path of **_D:\JDKs\17_**, you can however install it wherever and name the folders whatever you like.
6. After creating the folders and sub-folders you want click on the folder you want to install to and click the "**_OK_**" button
7. Click the "**_Next_**" button
8. Wait for it install
9. Click the "**_Close_**" button

> [!IMPORTANT]  
> If you selected a custom installation your system might not know that there is Java on your system and where it is, meaning Java apps still wont work. Please go to [Setting Environment Variables](#Environment)

#### Linux
<a name="JavaInstallLinux"></a>
Under Construction
#### MacOS
<a name="JavaInstallMacOS"></a>
Under Construction
### Setting Environment Variables
<a name="Environment"></a>

Environment Variables are variables stored in your system for apps to use. For example Java: windows stores the installation location of Java and then when you run a JAR file it looks for the location of Java by going to the Environment Variables and going to the location of Java specifically in the "Path" system variable.

To add Java to system variables follow these steps:

#### Windows
<a name="EnvironmentWindows"></a>
1. Search for "**_Environment_**" and click on "**_Edit the system environment variables_**"
2. Click the "**_Environment Variables_**" button on the bottom right of the new window
3. On the buttom half of this new windows, under "**_System_**" variables, find the "**_Path_**" variable and open it by double click it or clicking the "**_Edit..._**" button
4. On this new windows click the "**_New_**" button
5. In the textbox that appears enter the location you installed your java to. For example I installed it in **_D:\JDKs\17_**, so I would put **_D:\JDKs\17_** in the text box.
6. Click the "**_OK_**" button on all of the previous 3 windows that opened.

#### Linux
<a name="EnvironmentLinux"></a>
Under Construction
#### MacOS
<a name="EnvironmentMacOS"></a>
Under Construction

## GlobalQuake
<a name="GlobalQuake"></a>

### Download
<a name="GlobalQuakeDownload"></a>
Go to [GlobalQuakes releases page](https://github.com/xspanger3770/GlobalQuake/releases) and click the file named GlobalQuake_0.9.5.jar (the version number might be different).
You might get an error saying this file could be dangerous or malicious, however this isn't the case, click the button that confirms you want to continue with download.

### Installation
<a name="GlobalQuakeInstall"></a>
Move the jar file that you downloaded to the location you want it to be.\
Keeping it in a folder is recommended for organisation as it will create a new folder with data in it.\
If you do put it in a folder you can create a shortcut to the jar file for easier access.

### Running
<a name="GlobalQuakeRun"></a>
Run it by simply double clicking it or in a terminal or command line going to the directory it is located and using the command `java -jar [filename].jar`.
> [!NOTE]<br>
> You can add `-Xms8G -Xmx8G` parameters to specify the maximum amount of ram in gigabytes, replacing "8" with your own number.
> Experimental You can also improve your FPS by adding the parameter `-Dsun.java2d.opengl=True`.
