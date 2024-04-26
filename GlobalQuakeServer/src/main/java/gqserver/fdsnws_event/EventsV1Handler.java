package gqserver.fdsnws_event;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.URL;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;

import java.util.List;
import java.util.Map;

import org.tinylog.Logger;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import globalquake.core.archive.ArchivedQuake;
import globalquake.core.earthquake.EarthquakeDataExport;
import globalquake.core.exception.RuntimeApplicationException;

public class EventsV1Handler implements HttpHandler {
    @Override
    public void handle(HttpExchange exchange) throws IOException {
        HttpCatchAllLogger.logIncomingRequest(exchange);


        //check if application.wadl was requested
        if (exchange.getRequestURI().toString().endsWith("application.wadl")) {
            URL wadlURL = getClass().getClassLoader().getResource("fdsnws_event_application.wadl");

            if (wadlURL == null) {
                HttpRequestException ex = new HttpRequestException(500, "Internal Server Error");
                ex.transmitToClient(exchange);
                Logger.error(new RuntimeApplicationException("Wadl URL is null!"));
                return;
            }

            String wadl;

            try (InputStream in = wadlURL.openStream()) {
                wadl = new String(in.readAllBytes());
            } catch (Exception e) {
                Logger.error(e);
                HttpRequestException ex = new HttpRequestException(500, "Internal Server Error");
                ex.transmitToClient(exchange);
                return;
            }

            HttpResponse response = new HttpResponse(200, wadl, "application/xml");
            sendResponse(exchange, response);
            return;
        }

        //Parse the query string first to avoid extra work if there is an error
        FdsnwsEventsRequest request;
        try {
            request = new FdsnwsEventsRequest(exchange);
        } catch (HttpRequestException e) {
            //An error occurred parsing the query string
            e.transmitToClient(exchange);
            return;
        }


        //TODO: This gets every earthquake in the database. This relies on the database not being too large.
        List<ArchivedQuake> earthquakes = EarthquakeDataExport.getArchivedAndLiveEvents();

        List<ArchivedQuake> filteredQuakes = filterEventDataWithRequest(earthquakes, request);

        String formattedResult;
        String contentType;

        switch (request.format) {
            case "xml" -> {
                formattedResult = EarthquakeDataExport.getQuakeMl(filteredQuakes);
                contentType = "application/xml";
            }
            case "json", "geojson" -> {
                formattedResult = EarthquakeDataExport.getGeoJSON(filteredQuakes).toString();
                contentType = "application/json";
            }
            case "text" -> {
                formattedResult = EarthquakeDataExport.getText(filteredQuakes);
                contentType = "text/plain";
            }
            default -> {
                //This should never happen. This request should have been caught in the parameter checks
                //Don't Panic
                HttpRequestException e = new HttpRequestException(500, "Internal Server Error");
                e.transmitToClient(exchange);
                RuntimeApplicationException runtimeApplicationException = new RuntimeApplicationException("Somehow a point was reached that should have been unreachable. If code was just changed, that is the problem.");
                //No need to pass this to the error handler, just log it.
                Logger.error(runtimeApplicationException);
                return;
            }
        }

        //If there are no earthquakes, then set the response code to the nodata code
        int responseCode = !filteredQuakes.isEmpty() ? 200 : request.nodata;

        HttpResponse response = new HttpResponse(responseCode, formattedResult, contentType);
        sendResponse(exchange, response);
    }


    private List<ArchivedQuake> filterEventDataWithRequest(List<ArchivedQuake> earthquakes, FdsnwsEventsRequest request) {
        //Order does not necessarily matter here.

        //A basic wrapper is made to easily filter the earthquakes
        class protoquake {
            public ArchivedQuake quake;
            public boolean include;
        }
        List<protoquake> protoquakes = new ArrayList<>();
        for (ArchivedQuake quake : earthquakes) {
            protoquake protoquake = new protoquake();
            protoquake.quake = quake;
            protoquake.include = true;
            protoquakes.add(protoquake);
        }

        //If an event falls outside of any of the filters, it is removed from the list
        for (protoquake protoquake : protoquakes) {
            //Filter by time
            if (protoquake.quake.getOrigin() < request.starttime.getTime()) {
                protoquake.include = false;
            }
            if (protoquake.quake.getOrigin() > request.endtime.getTime()) {
                protoquake.include = false;
            }

            //Filter by latitude
            if (protoquake.quake.getLat() < request.minlatitude) {
                protoquake.include = false;
            }
            if (protoquake.quake.getLat() > request.maxlatitude) {
                protoquake.include = false;
            }

            //Filter by longitude
            if (protoquake.quake.getLon() < request.minlongitude) {
                protoquake.include = false;
            }
            if (protoquake.quake.getLon() > request.maxlongitude) {
                protoquake.include = false;
            }

            //Filter by depth
            if (protoquake.quake.getDepth() < request.mindepth) {
                protoquake.include = false;
            }
            if (protoquake.quake.getDepth() > request.maxdepth) {
                protoquake.include = false;
            }

            //Filter by magnitude
            if (protoquake.quake.getMag() < request.minmagnitude) {
                protoquake.include = false;
            }
            if (protoquake.quake.getMag() > request.maxmagnitude) {
                protoquake.include = false;
            }
        }

        //Remove all earthquakes that are not included
        protoquakes.removeIf(protoquake -> !protoquake.include);

        //Convert the protoquakes back into a list of ArchivedQuakes
        List<ArchivedQuake> filteredEarthquakes = new ArrayList<>();
        for (protoquake protoquake : protoquakes) {
            filteredEarthquakes.add(protoquake.quake);
        }

        return filteredEarthquakes;


    }

    private static void sendResponse(HttpExchange exchange, HttpResponse response) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", response.responseContentType());
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*"); //TODO: make this configurable
        exchange.sendResponseHeaders(response.responseCode(), response.responseContent().length());
        OutputStream os = exchange.getResponseBody();
        os.write(response.responseContent().getBytes());
        os.close();
    }


    private record HttpResponse(int responseCode, String responseContent, String responseContentType) {
    }

    @SuppressWarnings("unused")
    private static class FdsnwsEventsRequest {
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

        public FdsnwsEventsRequest(HttpExchange exchange) throws HttpRequestException {
            initDefaultParameters();
            parseQuery(exchange.getRequestURI());
        }

        private void parseQuery(URI uri) throws HttpRequestException {
            //Comments here appear once when needed to avoid duplicated explanations
            Map<String, String> parameters = parseQueryString(uri.getQuery());

            String start1 = parameters.get("start");
            String start2 = parameters.get("starttime");
            //If both are null, then the default is used
            if (start1 != null) {
                starttime = EventsV1ParamChecks.parseDate(start1);
            } else if (start2 != null) {
                starttime = EventsV1ParamChecks.parseDate(start2);
            }
            //If an error occurred, tell the user and log it
            if (starttime == null) {
                throw new HttpRequestException(400, "Issue parsing start time. Use the format \"YYYY-MM-DDTHH:MM:SS\" UTC time");
            }

            String end1 = parameters.get("end");
            String end2 = parameters.get("endtime");
            if (end1 != null) {
                endtime = EventsV1ParamChecks.parseDate(end1);
            } else if (end2 != null) {
                endtime = EventsV1ParamChecks.parseDate(end2);
            }
            if (endtime == null) {
                throw new HttpRequestException(400, "Issue parsing end time. Use the format of \"YYYY-MM-DDTHH:MM:SS\" UTC time");
            }

            String minlat1 = parameters.get("minlat");
            String minlat2 = parameters.get("minimumlatitude");
            if (minlat1 != null) {
                minlatitude = EventsV1ParamChecks.parseLatitude(minlat1);
            } else if (minlat2 != null) {
                minlatitude = EventsV1ParamChecks.parseLatitude(minlat2);
            }
            if (minlatitude == null) {
                throw new HttpRequestException(400, "Issue parsing minimum latitude. Make sure it is between -90 and 90");
            }

            String maxlat1 = parameters.get("maxlat");
            String maxlat2 = parameters.get("maximumlatitude");
            if (maxlat1 != null) {
                maxlatitude = EventsV1ParamChecks.parseLatitude(maxlat1);
            } else if (maxlat2 != null) {
                maxlatitude = EventsV1ParamChecks.parseLatitude(maxlat2);
            }
            if (maxlatitude == null) {
                throw new HttpRequestException(400, "Issue parsing maximum latitude. Make sure it is between -90 and 90");
            }

            String minlon1 = parameters.get("minlon");
            String minlon2 = parameters.get("minimumlongitude");
            if (minlon1 != null) {
                minlongitude = EventsV1ParamChecks.parseLongitude(minlon1);
            } else if (minlon2 != null) {
                minlongitude = EventsV1ParamChecks.parseLongitude(minlon2);
            }
            if (minlongitude == null) {
                throw new HttpRequestException(400, "Issue parsing minimum longitude. Make sure it is between -180 and 180");
            }

            String maxlon1 = parameters.get("maxlon");
            String maxlon2 = parameters.get("maximumlongitude");
            if (maxlon1 != null) {
                maxlongitude = EventsV1ParamChecks.parseLongitude(maxlon1);
            } else if (maxlon2 != null) {
                maxlongitude = EventsV1ParamChecks.parseLongitude(maxlon2);
            }
            if (maxlongitude == null) {
                throw new HttpRequestException(400, "Issue parsing maximum longitude. Make sure it is between -180 and 180");
            }

            String lat1 = parameters.get("lat");
            String lat2 = parameters.get("latitude");
            if (lat1 != null) {
                latitude = EventsV1ParamChecks.parseLatitude(lat1);
            } else if (lat2 != null) {
                latitude = EventsV1ParamChecks.parseLatitude(lat2);
            }
            //Either lat was null and the result is null
            if ((lat1 != null || lat2 != null) && latitude == null) {
                throw new HttpRequestException(400, "Issue parsing latitude. Make sure it is between -90 and 90");
            }

            String lon1 = parameters.get("lon");
            String lon2 = parameters.get("longitude");
            if (lon1 != null) {
                longitude = EventsV1ParamChecks.parseLongitude(lon1);
            } else if (lon2 != null) {
                longitude = EventsV1ParamChecks.parseLongitude(lon2);
            }
            if ((lon1 != null || lon2 != null) && longitude == null) {
                throw new HttpRequestException(400, "Issue parsing longitude. Make sure it is between -180 and 180");
            }

            //minradius
            //maxradius

            String mindepth1 = parameters.get("mindepth");
            if (mindepth1 != null) {
                mindepth = EventsV1ParamChecks.parseDepth(mindepth1);
            }
            if (mindepth == null) {
                throw new HttpRequestException(400, "Issue parsing minimum depth");
            }

            String maxdepth1 = parameters.get("maxdepth");
            if (maxdepth1 != null) {
                maxdepth = EventsV1ParamChecks.parseDepth(maxdepth1);
            }
            if (maxdepth == null) {
                throw new HttpRequestException(400, "Issue parsing maximum depth");
            }

            String minmag1 = parameters.get("minmag");
            String minmag2 = parameters.get("minmagnitude");
            if (minmag1 != null) {
                minmagnitude = EventsV1ParamChecks.parseMagnitude(minmag1);
            } else if (minmag2 != null) {
                minmagnitude = EventsV1ParamChecks.parseMagnitude(minmag2);
            }
            if (minmagnitude == null) {
                throw new HttpRequestException(400, "Issue parsing minimum magnitude, make sure it is between -10 and 10");
            }

            String maxmag1 = parameters.get("maxmag");
            String maxmag2 = parameters.get("maxmagnitude");
            if (maxmag1 != null) {
                maxmagnitude = EventsV1ParamChecks.parseMagnitude(maxmag1);
            } else if (maxmag2 != null) {
                maxmagnitude = EventsV1ParamChecks.parseMagnitude(maxmag2);
            }
            if (maxmagnitude == null) {
                throw new HttpRequestException(400, "Issue parsing maximum magnitude, make sure it is between -10 and 10");
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
            if (format1 != null) {
                //This might throw an exception
                format = EventsV1ParamChecks.parseFormat(format1);
            }
            if (format == null) {
                throw new HttpRequestException(400, "Issue parsing format. Make sure it is one of \"xml\", \"json\", \"geojson\", or \"text\"");
            }

            String nodata1 = parameters.get("nodata");
            if (nodata1 != null) {
                nodata = EventsV1ParamChecks.parseNoData(nodata1);
            }
            if (nodata == 0) {
                throw new HttpRequestException(400, "Issue parsing nodata. Make sure it is between 1 and 999");
            }

        }


        private void initDefaultParameters() {
            //Required parameters are set to reasonable defaults
            starttime = new Date(System.currentTimeMillis() - 3600 * 1000); //one hour ago
            endtime = new Date(System.currentTimeMillis()); //now

            //entire world
            minlatitude = -90f;
            maxlatitude = 90f;
            minlongitude = -180f;
            maxlongitude = 180f;

            //Anything outside of this is impossible
            float earth_radius = 6371f; //km

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

    @SuppressWarnings("unused")
    public static class HttpRequestException extends Exception {
        private final int errorCode;
        private final String errorMessage;
        private boolean revealError = true;

        public HttpRequestException(int errorCode, String errorMessage) {
            this.errorCode = errorCode;
            this.errorMessage = errorMessage;
        }

        public int getErrorCode() {
            return errorCode;
        }

        public String getErrorMessage() {
            return errorMessage;
        }

        public boolean shouldRevealError() {
            return revealError;
        }

        public void setRevealError(boolean revealError) {
            this.revealError = revealError;
        }

        public void transmitToClient(HttpExchange exchange) throws IOException {
            HttpResponse response;
            if (!revealError) {
                response = new HttpResponse(500, "Internal Server Error", "text/plain");
                sendResponse(exchange, response);
            }

            response = new HttpResponse(errorCode, errorMessage, "text/plain");
            sendResponse(exchange, response);
        }

    }
}
