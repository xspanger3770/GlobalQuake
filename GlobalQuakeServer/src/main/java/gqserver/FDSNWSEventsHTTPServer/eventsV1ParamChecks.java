package gqserver.FDSNWSEventsHTTPServer;

import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

import gqserver.FDSNWSEventsHTTPServer.EventsV1Handler.HttpRequestException;


public class EventsV1ParamChecks {
    //2011-03-11T05:00:00 UTC
    static public Date parseDate(String date){
        //Takes a string in the format of "YYYY-MM-DDTHH:MM:SS" UTC time and returns a Date object
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss");
        dateFormat.setTimeZone(java.util.TimeZone.getTimeZone("UTC"));
        Date parsedDate = null;
        try{
            parsedDate = dateFormat.parse(date);
        }catch(Exception e){} 
        return parsedDate;
    }

    //A bunch of fancy parseFloats that check for stuff

    static public Float parseLatitude(String latitude){
        //Takes a string in the format of "[-]DD.DD" and returns a Float object
        Float parsedLatitude = null;
        try{
            parsedLatitude = Float.parseFloat(latitude);
        }catch(Exception e){}
        if(parsedLatitude > 90 || parsedLatitude < -90){
            return null;
        }
        return parsedLatitude;
    }

    //radius

    static public Float parseLongitude(String longitude){
        //Takes a string in the format of "[-]DDD.DD" and returns a Float object
        Float parsedLongitude = null;
        try{
            parsedLongitude = Float.parseFloat(longitude);
        }catch(Exception e){} 
        if(parsedLongitude > 180 || parsedLongitude < -180){
            return null;
        }
        return parsedLongitude;
    }

    static public Float parseDepth(String depth){
        /*Takes a string and makes sure it is a valid depth
         *In the future, this will need an appropriate range check
        */
        Float parsedDepth = null;
        try{
            parsedDepth = Float.parseFloat(depth);
        }catch(Exception e){} 
        return parsedDepth;
    }

    static public Float parseMagnitude(String magnitude){
        //Takes a string and makes sure it is a valid magnitude
        Float parsedMagnitude = null;
        try{
            parsedMagnitude = Float.parseFloat(magnitude);
        }catch(Exception e){}
        if(parsedMagnitude > 10 || parsedMagnitude < -10){
            return null;
        }
        return parsedMagnitude;
    }

    //several other things

    static public String parseFormat(String format) throws HttpRequestException{
        //Takes a string and makes sure it is a valid format
        List<String> validFormats = Arrays.asList("quakeml", "geojson", "text", "json", "xml");
        List<String> disallowedDroids = Arrays.asList("xmlp", "geojsonp", "jsonp", "quakemlp", "textp");

        if(validFormats.contains(format)){
            return format;
        }

        if(disallowedDroids.contains(format)){
            throw new HttpRequestException(400, "Invalid format. The format " + format + " are not the droids you're looking for");
        }

        //If we get here, the format is invalid
        return null;

    }

    static public int parseNoData(String noData){
        //Takes a string and makes sure it is a valid nodata
        int parsedNoData = 0;
        try{
            parsedNoData = Integer.parseInt(noData);
        }catch(Exception e){}
        if(parsedNoData > 999 || parsedNoData < 1){
            return 0;
        }
        return parsedNoData;
    }
}
