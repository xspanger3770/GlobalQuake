package com.morce.globalquake.utils;

import java.io.Serializable;

public class Level implements Serializable {
    public static final long serialVersionUID = 4362L;

    private final String name;
    private final double pga;
    private final int index;

    public Level(String name, double pga, int index){
        this.name=name;
        this.pga=pga;
        this.index=index;
    }

    public int getIndex() {
        return index;
    }

    public String getName() {
        return name;
    }

    public double getPga() {
        return pga;
    }
}
