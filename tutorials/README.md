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
> If you're on a recent MacOS device you should make sure of your architecture since some recent MacOS devices use arm64.

### Downloading
<a name="JavaDownload"></a>
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
For Linux it highly depends on the distribution you are running. If you are running Debian, or a distribution based on Debian like Ubuntu ([A list of Debian based distros](https://en.wikipedia.org/wiki/List_of_Linux_distributions#Debian-based)), download the "**_x64 Debian Package_**". If your distro is based off of Redhat ([A list of Redhat based distros](https://en.wikipedia.org/wiki/List_of_Linux_distributions#RPM-based)) use the "**_x64 RPM Package_**".

#### MacOS
<a name="JavaDownloadMacOS"></a>
For MacOS download the "**_x64 DMG Installer_**".

### Installing
<a name="JavaInstall"></a>

#### Windows
<a name="JavaInstallWindows"></a>
1. Download the installer <br> ![1](https://github.com/CentreMetre/GlobalQuake/assets/61330791/f03fee45-0d24-4639-90e3-0b764d9a1c88)
2. Open the installer <br> ![2](https://github.com/CentreMetre/GlobalQuake/assets/61330791/73733a36-1aeb-44f7-95e5-787bfc2b2a54)
3. Click the "**_Next_**" button on the first Screen <br> ![3](https://github.com/CentreMetre/GlobalQuake/assets/61330791/db959e76-b6a0-4456-b1f8-3f93ad5262a4)
4. Choosing a custom install location (if you don't want a custom install location skip to step [7](#JavaInstallWindowsStep7) <br> ![4](https://github.com/CentreMetre/GlobalQuake/assets/61330791/9892291c-c544-4562-ad49-ef5f5cfdd1c2)
5. Go to the location you want to install Java with the file explorer. You can choose another drive by clicking the topdown menu at the top with the little downwards arrow on it. You can add folders using the button on the top right with the folder and shine icon. For example I went to my D drive and created a folder called JDKs and a sub folder called 17, for the JDK 17 version, for the path of "**_D:\JDKs\17_**", you can however install it wherever and name the folders whatever you like. <br> ![5 1](https://github.com/CentreMetre/GlobalQuake/assets/61330791/a1572ff9-452f-4dc8-bc62-3a765dd662cd)
6. After creating the folders and sub-folders you want click on the folder you want to install to and click the "**_OK_**" button <br> ![5](https://github.com/CentreMetre/GlobalQuake/assets/61330791/77fd9f61-926c-44f9-82ed-48efe21e3cdc)
<a name="JavaInstallWindowsStep7"></a>
7. Click the "**_Next_**" button <br> ![6](https://github.com/CentreMetre/GlobalQuake/assets/61330791/6ae625da-a3c6-4a10-a2e9-febde88c1a86)
8. Wait for it install <br> ![7](https://github.com/CentreMetre/GlobalQuake/assets/61330791/fdaf1359-5731-49fe-a5a1-c9836630b11a)
9. Click the "**_Close_**" button

> [!IMPORTANT]  
> If you selected a custom installation your system might not know that there is Java on your system and where it is, meaning Java apps still wont work. Please go to [Setting Environment Variables](#Environment)

#### Linux
<a name="JavaInstallLinux"></a>

**APT (Ubuntu)**
1. Download the appropriate Java package. If you're not sure which one to download refer to [Java Linux Download](#JavaDownloadLinux). <br>![Screenshot from 2023-09-08 16-15-10](https://github.com/CentreMetre/GlobalQuake/assets/61330791/f17d1c85-5b5e-4c54-b3ea-a48ea8cb21c7)
2. Open the downloaded file with a package manager. You can either set the defualt app for .deb files to a package manager or just open this file with a package manager this time. To open with a package manager this one time right click the downloaded .deb file, click "**_Open With Another Application_**", then click software install and then select. <br> ![Screenshot from 2023-09-08 16-17-09](https://github.com/CentreMetre/GlobalQuake/assets/61330791/af7dfbea-7ee1-4dc1-aaa2-79c008ddafc2)
3. When your package manager opens click the install button and wait for it to install. If it asks for authentication press "**_OK_**" or "**_Authenticate_**", or if it asks for a password input your password and press the confirmation button. <br> ![Screenshot from 2023-09-08 16-18-13](https://github.com/CentreMetre/GlobalQuake/assets/61330791/4b00a5aa-314d-4b09-9619-01691d6c6e52)
4. Verify it installed succsefully by going into the terminal and entering ```java -version```. If you get something similar to the screenshot it has successfully installed. If you get an error saying ```bash: /usr/bin/java: No such file or directory``` it means it hasnt installed properly. Please try the steps again from step 1. <br> ![Screenshot from 2023-09-08 16-19-02](https://github.com/CentreMetre/GlobalQuake/assets/61330791/36c00f02-2c11-447a-9468-69846f84f1d4)

> [!NOTE] <br>
> Some buttons may be in different places or the UI may be entirly different depending on what desktop environment your OS is using. This tutorial was done using Ubuntu 22.04 with the GNOME desktop environment. 

**RPM (Fedora)**
1. Open the terminal.
2. Enter the command ```sudo rpm -i https://download.oracle.com/java/17/latest/jdk-17_linux-x64_bin.rpm```. It may ask for a password, if it does enter the password in the terminal. When you're typing it may look like nothing is happening, this is just to prevent other people from seeing your password, anything being typed is still going into the terminal. The terminal may look like it frozen for a while, it's not, it is just installing Java.<br> ![image](https://github.com/CentreMetre/GlobalQuake/assets/61330791/6b9e8b91-ef7f-42ec-aaf6-94ff6841d045)
3. Enter the command ```java -version``` to verify it has isntalled. If it is similar to the screen shot then it as successfully installed, if you get an error it might mean it hasnt installed properly, try again from step 1. <br> ![image](https://github.com/CentreMetre/GlobalQuake/assets/61330791/afc64581-c146-40b8-800f-c5f8a78f96e9)

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
3. On the buttom half of this new window, under "**_System_**" variables, find the "**_Path_**" variable and open it by double click it or clicking the "**_Edit..._**" button
4. On this new window click the "**_New_**" button
5. In the textbox that appears enter the location you installed your java to. For example I installed it in "**_D:\JDKs\17_**", so I would put "**_D:\JDKs\17_**" in the text box.
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
In Windows you can run it by simply double clicking the jar file, or in a Linux Distro (ie. Ubuntu) terminal or Windows command line by going to the directory it is located and using the command `java -jar [filename].jar`.
> [!NOTE]<br>
> You can add `-Xms8G -Xmx8G` parameters to specify the maximum amount of ram in gigabytes, replacing "8" with your own number.
> Experimental: You can also improve your FPS by adding the parameter `-Dsun.java2d.opengl=True`.

For example the full teminal or command line to start GlobalQuake (assuming you have already navigated to the appropriate folder) should look something like this:

> [!TIP]
> This Won't Work with a shortcut (on windows)

\
*Simple*\
`$ java -jar GlobalQuake0.10.x_pre10.jar`\
or\
*With The Extra Variables Added* (remember the instructions in the note above)\
`$ java -jar -Xms8G -Xmx8G -Dsun.java2d.opengl=true GlobalQuake0.10.x_pre10.jar`

> [!NOTE]
> ONLY work with the Minimum RAM needed as you can stall the system and cause a ton of complications by setting the `-Xms_G` and `-Xmx_G` numbers too high.\
> for instance if you have 8GB of RAM in you system the OS needs a minimum of 50% to work so the `-Xms_G` and `-Xmx_G` should not be higher than 4.\
> **Examples** `-Xms4G -Xmx4G` would be ok to run GlobalQuake Locally with a total of 8GB of ram.
