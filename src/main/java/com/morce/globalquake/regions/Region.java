package com.morce.globalquake.regions;

import java.awt.geom.Path2D;
import java.util.ArrayList;

import org.geojson.Polygon;

public class Region {

	private ArrayList<Path2D.Double> paths;
	private String name;
	private ArrayList<Polygon> raws;

	public Region(String name, ArrayList<Path2D.Double> paths, ArrayList<Polygon> raws) {
		this.paths = paths;
		this.raws = raws;
		this.name = name;
	}

	public ArrayList<Path2D.Double> getPaths() {
		return paths;
	}
	
	public ArrayList<Polygon> getRaws() {
		return raws;
	}

	public String getName() {
		return name;
	}

}
