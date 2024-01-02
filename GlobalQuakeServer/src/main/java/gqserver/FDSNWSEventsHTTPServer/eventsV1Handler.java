package gqserver.FDSNWSEventsHTTPServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;
import java.net.URL;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.geometry.spherical.oned.Arc;
import org.json.JSONArray;
import org.json.JSONObject;
import org.tinylog.Logger;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import globalquake.core.archive.ArchivedQuake;
import globalquake.core.earthquake.EarthquakeDataExport;
import globalquake.core.exception.RuntimeApplicationException;
import gqserver.FDSNWSEventsHTTPServer.HTTPCatchAllLogger;
import gqserver.FDSNWSEventsHTTPServer.eventsV1ParamChecks;

public class eventsV1Handler implements HttpHandler{
    @Override
    public void handle(HttpExchange exchange) throws IOException {
        HTTPCatchAllLogger.logIncomingRequest(exchange);



        //check if application.wadl was requested
        if(exchange.getRequestURI().toString().endsWith("application.wadl")){
            URL wadlURL = getClass().getClassLoader().getResource("fdsnws_event_application.wadl");
            String wadl = "";
            try{
                wadl = new String(wadlURL.openStream().readAllBytes());
            }catch(Exception e){
                Logger.error(e);
                HTTPRequestException ex = new HTTPRequestException(500, "Internal Server Error");
                ex.transmitToClient(exchange);
                return;
            }

            HTTPResponse response = new HTTPResponse(200, wadl, "application/xml");
            sendResponse(exchange, response);
            return;
        }

        //Parse the query string first to avoid extra work if there is an error
        FDSNWSEventsRequest request = null;
        try{
            request = new FDSNWSEventsRequest(exchange);
        }catch(HTTPRequestException e){
            //An error occurred parsing the query string
            e.transmitToClient(exchange);
            return;
        }


        //TODO: This gets every earthquake in the database. This relies on the database not being too large.
        List<ArchivedQuake> earthquakes = EarthquakeDataExport.getArchivedAndLiveEvents();

        List<ArchivedQuake> filteredQuakes = filterEventDataWithRequest(earthquakes, request);
        
        String formattedResult="";
        String contentType = "";

        if(request.format.equals("xml")){
            formattedResult = EarthquakeDataExport.GetQuakeMl(filteredQuakes);
            contentType = "application/xml";
        }
        else if(request.format.equals("json") || request.format.equals("geojson")){
            formattedResult = EarthquakeDataExport.getGeoJSON(filteredQuakes).toString();
            contentType = "application/json";
        }
        else if(request.format.equals("text")){
            formattedResult = EarthquakeDataExport.getText(filteredQuakes);
            contentType = "text/plain";
        }
        else{
            //This should never happen. This request should have been caught in the parameter checks
            //Don't Panic
            HTTPRequestException e = new HTTPRequestException(500, "Internal Server Error");
            e.transmitToClient(exchange);
            RuntimeApplicationException runtimeApplicationException = new RuntimeApplicationException("Somehow a point was reached that should have been unreachable. If code was just changed, that is the problem.");
            //No need to pass this to the error handler, just log it.
            Logger.error(runtimeApplicationException);
            return;
        }

        //If there are no earthquakes, then set the response code to the nodata code
        int responseCode = filteredQuakes.size() > 0 ? 200 : request.nodata;

        HTTPResponse response = new HTTPResponse(responseCode, formattedResult, contentType);
        sendResponse(exchange, response);
    }


    private List<ArchivedQuake> filterEventDataWithRequest(List<ArchivedQuake> earthquakes, FDSNWSEventsRequest request){
        //Order does not necessarily matter here.

        //A basic wrapper is made to easily filter the earthquakes
        class protoquake {public ArchivedQuake quake; public boolean include;}
        List<protoquake> protoquakes = new ArrayList<>();
        for(ArchivedQuake quake : earthquakes){
            protoquake protoquake = new protoquake();
            protoquake.quake = quake;
            protoquake.include = true;
            protoquakes.add(protoquake);
        }
        
        //If an event falls outside of any of the filters, it is removed from the list
        for(protoquake protoquake : protoquakes){
            //Filter by time
            if(protoquake.quake.getOrigin() < request.starttime.getTime()){
                protoquake.include = false;
            }
            if(protoquake.quake.getOrigin() > request.endtime.getTime()){
                protoquake.include = false;
            }

            //Filter by latitude
            if(protoquake.quake.getLat() < request.minlatitude){
                protoquake.include = false;
            }
            if(protoquake.quake.getLat() > request.maxlatitude){
                protoquake.include = false;
            }

            //Filter by longitude
            if(protoquake.quake.getLon() < request.minlongitude){
                protoquake.include = false;
            }
            if(protoquake.quake.getLon() > request.maxlongitude){
                protoquake.include = false;
            }

            //Filter by depth
            if(protoquake.quake.getDepth() < request.mindepth){
                protoquake.include = false;
            }
            if(protoquake.quake.getDepth() > request.maxdepth){
                protoquake.include = false;
            }

            //Filter by magnitude
            if(protoquake.quake.getMag() < request.minmagnitude){
                protoquake.include = false;
            }
            if(protoquake.quake.getMag() > request.maxmagnitude){
                protoquake.include = false;
            }
        }

        //Remove all earthquakes that are not included
        protoquakes.removeIf(protoquake -> !protoquake.include);

        //Convert the protoquakes back into a list of ArchivedQuakes
        List<ArchivedQuake> filteredEarthquakes = new ArrayList<>();
        for(protoquake protoquake : protoquakes){
            filteredEarthquakes.add(protoquake.quake);
        }

        return filteredEarthquakes;


    }

    private static void sendResponse(HttpExchange exchange, HTTPResponse response) throws IOException{
        exchange.getResponseHeaders().set("Content-Type", response.getResponseContentType());
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*"); //TODO: make this configurable
        exchange.sendResponseHeaders(response.getResponseCode(), response.getResponseContent().length());
        OutputStream os = exchange.getResponseBody();
        os.write(response.getResponseContent().getBytes());
        os.close();
    }


    private static class HTTPResponse{
        private int responseCode;
        private String responseContent;
        private String responseContentType;

        public HTTPResponse(int responseCode, String responseContent, String responseContentType){
            this.responseCode = responseCode;
            this.responseContent = responseContent;
            this.responseContentType = responseContentType;
        }

        public int getResponseCode(){
            return responseCode;
        }

        public String getResponseContent(){
            return responseContent;
        }

        public String getResponseContentType(){
            return responseContentType;
        }
    }

    private class FDSNWSEventsRequest {
        //start
        private Date starttime;            //Limit to events on or after the specified start time.
        //end
        private Date endtime;              //Limit to events on or before the specified end time.
        //minlat
        private Float minlatitude;        //Limit to events with a latitude larger than or equal to the specified minimum.
        //maxlat
        private Float maxlatitude;        //Limit to events with a latitude smaller than or equal to the specified maximum
        //minlon
        private Float minlongitude;       //Limit to events with a longitude larger than or equal to the specified minimum.
        //maxlon
        private Float maxlongitude;       //Limit to events with a longitude smaller than or equal to the specified maximum.
        //lat
        private Float latitude;           //Specify the latitude to be used for a radius search.
        //lon
        private Float longitude;             //Specify the longitude to be used for a radius search.
        private Float minradius;             //Limit to events within the specified minimum number of degrees from the geographic point defined by the latitude and longitude parameters.
        private Float maxradius;             //Limit to events within the specified maximum number of degrees from the geographic point defined by the latitude and longitude parameters.
        private Float mindepth;              //Limit to events with depth more than the specified minimum.
        private Float maxdepth;              //Limit to events with depth less than the specified maximum.
        //minmag
        private Float minmagnitude;          //Limit to events with a magnitude larger than the specified minimum.
        //maxmag
        private Float maxmagnitude;          //Limit to events with a magnitude smaller than the specified maximum.
        //magtype
        private String magnitudetype;         //Specify a magnitude type to use for testing the minimum and maximum limits.
        private String eventtype;             //Limit to events with a specified eventType. The parameter value can be a single item, a comma-separated list of items. Allowed values are from QuakeML or unknown if eventType is not given.
        private boolean includeallorigins;    //Specify if all origins for the event should be included, default is data center dependent but is suggested to be the preferred origin only.
        private boolean includeallmagnitudes; //Specify if all magnitudes for the event should be included, default is data center dependent but is suggested to be the preferred magnitude only.
        private boolean includearrivals;      //Specify if phase arrivals should be included.
        private String eventid;               //Select a specific event by ID; event identifiers are data center specific
        private int limit;                    //Limit the results to the specified number of events.
        private int offset;                   //Return results starting at the event count specified, starting at 1.
        
        private String orderby;             //Order the results. The allowed values are:
                                            //time - the default, order by origin descending time
                                            //time-asc - order by origin ascending time
                                            //magnitude - order by descending magnitude
                                            //magnitude-asc - order by ascending magnitude

        private String catalog;            //Limit to events from a specified catalog.
        private String contributor;        //Limit to events contributed by a specified contributor.
        private Date updatedafter;         //Limit to events updated after the specified time.
                                           //* While this option is not required it is highly recommended due to usefulness.

        private String format;             //Specify the output format. XML, json|geojson, text.
        private int nodata;                //HTTP status code to return when no data is found. Default is 204, no content.
        

        /*Create list of not implemented parameters
        public Set<String> notImplementedParameters = new HashSet<>();
        private void setNotImplementedParameters(){
            notImplementedParameters.add("lat");
            notImplementedParameters.add("latitude");
            notImplementedParameters.add("lon");
            notImplementedParameters.add("longitude");
            notImplementedParameters.add("minradius");
            notImplementedParameters.add("maxradius");
            notImplementedParameters.add("magtype");
            notImplementedParameters.add("magnitudetype");
            notImplementedParameters.add("eventtype");
            notImplementedParameters.add("includeallorigins");
            notImplementedParameters.add("includeallmagnitudes");
            notImplementedParameters.add("includearrivals");
            notImplementedParameters.add("eventid");
            notImplementedParameters.add("limit");
            notImplementedParameters.add("offset");
            notImplementedParameters.add("orderby");
            notImplementedParameters.add("catalog");
            notImplementedParameters.add("contributor");
            notImplementedParameters.add("updatedafter");
        }
        */

        public FDSNWSEventsRequest(HttpExchange exchange) throws HTTPRequestException{
            initDefaultParameters();
            ParseQuery(exchange.getRequestURI());
        }

        private void ParseQuery(URI uri) throws HTTPRequestException{
            //Comments here appear once when needed to avoid duplicated explanations
            Map<String, String> parameters = parseQueryString(uri.getQuery());

            String start1 = parameters.get("start");
            String start2 = parameters.get("starttime");
            //If both are null, then the default is used
            if(start1 != null){
                starttime = eventsV1ParamChecks.ParseDate(start1);
            }else if(start2 != null){
                starttime = eventsV1ParamChecks.ParseDate(start2);
            }
            //If an error occurred, tell the user and log it
            if(starttime == null){
                throw new HTTPRequestException(400, "Issue parsing start time. Use the format \"YYYY-MM-DDTHH:MM:SS\" UTC time");
            }

            String end1 = parameters.get("end");
            String end2 = parameters.get("endtime");
            if(end1 != null){
                endtime = eventsV1ParamChecks.ParseDate(end1);
            }else if(end2 != null){
                endtime = eventsV1ParamChecks.ParseDate(end2);
            }
            if(endtime == null){
                throw new HTTPRequestException(400, "Issue parsing end time. Use the format of \"YYYY-MM-DDTHH:MM:SS\" UTC time");
            }

            String minlat1 = parameters.get("minlat");
            String minlat2 = parameters.get("minimumlatitude");
            if(minlat1 != null){
                minlatitude = eventsV1ParamChecks.ParseLatitude(minlat1);
            }else if(minlat2 != null){
                minlatitude = eventsV1ParamChecks.ParseLatitude(minlat2);
            }
            if(minlatitude == null){
                throw new HTTPRequestException(400, "Issue parsing minimum latitude. Make sure it is between -90 and 90");
            }

            String maxlat1 = parameters.get("maxlat");
            String maxlat2 = parameters.get("maximumlatitude");
            if(maxlat1 != null){
                maxlatitude = eventsV1ParamChecks.ParseLatitude(maxlat1);
            }else if(maxlat2 != null){
                maxlatitude = eventsV1ParamChecks.ParseLatitude(maxlat2);
            }
            if(maxlatitude == null){
                throw new HTTPRequestException(400, "Issue parsing maximum latitude. Make sure it is between -90 and 90");
            }

            String minlon1 = parameters.get("minlon");
            String minlon2 = parameters.get("minimumlongitude");
            if(minlon1 != null){
                minlongitude = eventsV1ParamChecks.ParseLongitude(minlon1);
            }else if(minlon2 != null){
                minlongitude = eventsV1ParamChecks.ParseLongitude(minlon2);
            }
            if(minlongitude == null){
                throw new HTTPRequestException(400, "Issue parsing minimum longitude. Make sure it is between -180 and 180");
            }

            String maxlon1 = parameters.get("maxlon");
            String maxlon2 = parameters.get("maximumlongitude");
            if(maxlon1 != null){
                maxlongitude = eventsV1ParamChecks.ParseLongitude(maxlon1);
            }else if(maxlon2 != null){
                maxlongitude = eventsV1ParamChecks.ParseLongitude(maxlon2);
            }
            if(maxlongitude == null){
                throw new HTTPRequestException(400, "Issue parsing maximum longitude. Make sure it is between -180 and 180");
            }
 
            String lat1 = parameters.get("lat");
            String lat2 = parameters.get("latitude");
            if(lat1 != null){
                latitude = eventsV1ParamChecks.ParseLatitude(lat1);
            }else if(lat2 != null){
                latitude = eventsV1ParamChecks.ParseLatitude(lat2);
            }
            //Either lat was null and the result is null
            if( (lat1 != null || lat1 != null) && latitude == null){
                throw new HTTPRequestException(400, "Issue parsing latitude. Make sure it is between -90 and 90");
            }

            String lon1 = parameters.get("lon");
            String lon2 = parameters.get("longitude");
            if(lon1 != null){
                longitude = eventsV1ParamChecks.ParseLongitude(lon1);
            }else if(lon2 != null){
                longitude = eventsV1ParamChecks.ParseLongitude(lon2);
            }
            if( (lon1 != null || lon2 != null) && longitude == null){
                throw new HTTPRequestException(400, "Issue parsing longitude. Make sure it is between -180 and 180");
            }

            //minradius
            //maxradius

            String mindepth1 = parameters.get("mindepth");
            if(mindepth1 != null){
                mindepth = eventsV1ParamChecks.ParseDepth(mindepth1);
            }
            if(mindepth == null){
                throw new HTTPRequestException(400, "Issue parsing minimum depth");
            }

            String maxdepth1 = parameters.get("maxdepth");
            if(maxdepth1 != null){
                maxdepth = eventsV1ParamChecks.ParseDepth(maxdepth1);
            }
            if(maxdepth == null){
                throw new HTTPRequestException(400, "Issue parsing maximum depth");
            }

            String minmag1 = parameters.get("minmag");
            String minmag2 = parameters.get("minmagnitude");
            if(minmag1 != null){
                minmagnitude = eventsV1ParamChecks.ParseMagnitude(minmag1);
            }else if(minmag2 != null){
                minmagnitude = eventsV1ParamChecks.ParseMagnitude(minmag2);
            }
            if(minmagnitude == null){
                throw new HTTPRequestException(400, "Issue parsing minimum magnitude, make sure it is between -10 and 10");
            }

            String maxmag1 = parameters.get("maxmag");
            String maxmag2 = parameters.get("maxmagnitude");
            if(maxmag1 != null){
                maxmagnitude = eventsV1ParamChecks.ParseMagnitude(maxmag1);
            }else if(maxmag2 != null){
                maxmagnitude = eventsV1ParamChecks.ParseMagnitude(maxmag2);
            }
            if(maxmagnitude == null){
                throw new HTTPRequestException(400, "Issue parsing maximum magnitude, make sure it is between -10 and 10");
            }

            //magtype, magnitudetype
            //eventtype
            //includeallorigins
            //includeallmagnitudes
            //includearrivals
            //eventid
            //limit
            //offset
            //orderby
            //catalog
            //contributor
            //updatedafter
            
            String format1 = parameters.get("format");
            if(format1 != null){
                //This might throw an exception
                format = eventsV1ParamChecks.ParseFormat(format1);
            }
            if(format == null){
                throw new HTTPRequestException(400, "Issue parsing format. Make sure it is one of \"xml\", \"json\", \"geojson\", or \"text\"");
            }

            String nodata1 = parameters.get("nodata");
            if(nodata1 != null){
                nodata = eventsV1ParamChecks.ParseNoData(nodata1);
            }
            if(nodata == 0){
                throw new HTTPRequestException(400, "Issue parsing nodata. Make sure it is between 1 and 999");
            }

        }

        
        private void initDefaultParameters(){
            //Required parameters are set to reasonable defaults
            starttime = new Date(System.currentTimeMillis() - 3600 * 1000); //one hour ago
            endtime = new Date(System.currentTimeMillis()); //now

            //entire world
            minlatitude = -90f;
            maxlatitude = 90f;
            minlongitude = -180f;
            maxlongitude = 180f;


            //Anything outside of this is impossible
            Float earth_radius = 6371f; //km

            //any depth
            mindepth = -earth_radius;
            maxdepth = earth_radius;

            //any magnitude
            minmagnitude = -10f;
            maxmagnitude = 10f;

            //Default format is XML
            format = "xml";

            //nodata defaults to 204: no content
            nodata = 204;
        }

        private Map<String, String> parseQueryString(String queryString) {
            Map<String, String> parameters = new HashMap<>();
            if (queryString != null) {
                String[] pairs = queryString.split("&");
                for (String pair : pairs) {
                    String[] keyValue = pair.split("=");
                    if (keyValue.length == 2) {
                        String key = keyValue[0];
                        String value = keyValue[1];
                        parameters.put(key, value);
                    }
                }
            }
            return parameters;
        }
    }

    public static class HTTPRequestException extends Exception{
        private int errorCode;
        private String errorMessage;
        private boolean revealError = true;

        public HTTPRequestException(int errorCode, String errorMessage){
            this.errorCode = errorCode;
            this.errorMessage = errorMessage;
        }

        public int getErrorCode(){
            return errorCode;
        }

        public String getErrorMessage(){
            return errorMessage;
        }

        public boolean shouldRevealError(){
            return revealError;
        }

        public void setRevealError(boolean revealError){
            this.revealError = revealError;
        }

        public void transmitToClient(HttpExchange exchange) throws IOException{
            HTTPResponse response = null;
            if(!revealError){
                response = new HTTPResponse(500, "Internal Server Error", "text/plain");
                sendResponse(exchange, response);
            }

            response = new HTTPResponse(errorCode, errorMessage, "text/plain");
            sendResponse(exchange, response);
        }

    }
}